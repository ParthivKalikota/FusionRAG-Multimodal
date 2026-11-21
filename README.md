# FusionRAG-Multimodal

A multimodal Retrieval-Augmented Generation (RAG) project. This README explains the theory behind the system.

---

## 1. Background

Large Language Models (LLMs) generate fluent text but have two core limitations:

- **Static knowledge:** they only know what was in their training data.
- **Hallucination:** they can produce confident but incorrect statements.

**Retrieval-Augmented Generation (RAG)** addresses this by retrieving relevant external information at query time and giving it to the model as context, so answers are grounded in real documents.

## 2. Retrieval-Augmented Generation

A standard RAG pipeline has three stages:

1. **Indexing:** documents are split into chunks, converted into vectors (embeddings), and stored in a vector database.
2. **Retrieval:** the user query is embedded with the same model, and the most similar chunks are found using a similarity measure such as cosine similarity:

   ```
   sim(q, d) = (q · d) / (‖q‖ ‖d‖)
   ```

3. **Generation:** the retrieved chunks are added to the prompt, and the LLM generates an answer conditioned on them.

Formally, for a query *q* and corpus *D*, RAG models

```
p(r | q) = Σ_{d ∈ top-k(D, q)}  p(d | q) · p(r | q, d)
```

where *r* is the response and *d* are the retrieved documents.

## 3. Why Multimodal?

Real-world knowledge is rarely text-only. Documents contain **images, charts, tables, diagrams, and sometimes audio or video**. Text-only RAG loses this information. **Multimodal RAG** extends the pipeline so that:

- the **corpus** can contain multiple modalities (text, images, and audio converted to text),
- the **query** can be text, an image, or both,
- the **generator** is a Multimodal LLM (MLLM / Vision-Language Model) that can reason over retrieved text and images together.

### The cross-modal challenge

Text and images live in different representation spaces. To compare them, the system needs either:

- a **joint embedding space** (e.g. CLIP-style contrastive models that map text and images into the same vector space), or
- **modality-specific encoders** with a separate index per modality, combined later at the score level.

### Audio via speech-to-text (Whisper)

Audio (lectures, meetings, podcasts, video soundtracks) can't be searched directly by a text retriever. A common and effective strategy is to **convert audio into text first**, then treat the transcript like any other document.

**Whisper** is OpenAI's open-source automatic speech recognition (ASR) model. It is an encoder-decoder Transformer trained on a large, diverse set of multilingual audio paired with transcripts, which makes it robust to accents, background noise, and technical vocabulary. It also supports transcription in many languages and translation to English.

How it fits into the RAG pipeline:

1. **Extract audio:** pull the audio track from the file (e.g. from a video).
2. **Transcribe:** Whisper processes the audio in ~30-second windows (log-Mel spectrogram in, text tokens out) and produces a transcript, optionally with **timestamps**.
3. **Chunk:** the transcript is split into passages, ideally along segment boundaries so each chunk keeps its start and end time.
4. **Embed and index:** transcript chunks are embedded and stored alongside text chunks, with metadata (source file, timestamp range, modality = `audio`).
5. **Retrieve and generate:** at query time, audio-derived chunks compete with other chunks, and the timestamps let answers point back to the exact moment in the recording.

Why transcribe instead of embedding raw audio? Text embeddings are mature, cheap to search, and work directly with existing text retrievers and LLMs. The trade-off is that non-speech information (tone, music, sound events) is lost, and any ASR errors propagate into retrieval.

## 4. Fusion

"Fusion" means combining evidence from different retrievers or modalities into a single ranked result. Common approaches:

### 4.1 Early fusion (feature level)
Embeddings from different modalities are combined (e.g. concatenated or averaged) into one representation before retrieval.

### 4.2 Late fusion (score level)
Each modality or retriever produces its own ranked list, and the lists are merged afterwards.

**Weighted score fusion**

```
score(d) = α · s_text(d) + β · s_image(d)
```

where α and β control how much each modality contributes (scores are normalized first so they are comparable).

**Reciprocal Rank Fusion (RRF)**

```
RRF(d) = Σ_{r ∈ rankers}  1 / (k + rank_r(d))
```

RRF only uses rank positions, not raw scores, so it works well when scores from different retrievers are on different scales. *k* is a smoothing constant (commonly 60).

### 4.3 Hybrid retrieval
Combines **dense retrieval** (semantic embeddings, good at meaning) with **sparse retrieval** (BM25 / keyword matching, good at exact terms). Fusing both generally improves recall over either alone.

### 4.4 Re-ranking
After fusion, a stronger but slower model (e.g. a cross-encoder) re-scores the top candidates to improve precision before they reach the generator.

## 5. Generation

The fused, re-ranked context (text passages plus images or image descriptions) is placed into the prompt of a multimodal LLM. Good generation depends on:

- **Context selection:** only the top-k most relevant items, to stay within the context window and avoid noise.
- **Grounding instructions:** prompting the model to answer from the provided context and say when the answer is not present.
- **Cross-modal reasoning:** the model must align information across text and visuals (for example, reading a chart and its caption together).

## 6. Evaluation

**Retrieval quality**

| Metric | Meaning |
|---|---|
| Recall@k | Fraction of queries where a relevant item appears in the top-k |
| Precision@k | Fraction of the top-k that are relevant |
| MRR | Average of 1/rank of the first relevant result |
| nDCG@k | Ranking quality that rewards relevant items placed higher |

**Transcription quality (audio)**

- **Word Error Rate (WER):** `(substitutions + deletions + insertions) / words in reference`; lower is better.

**Generation quality**

- **Faithfulness / groundedness:** is the answer supported by the retrieved context?
- **Answer relevance:** does it actually address the question?
- **Correctness:** exact match / F1 against reference answers, or LLM-as-judge scoring.

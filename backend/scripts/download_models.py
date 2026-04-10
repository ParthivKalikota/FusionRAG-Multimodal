"""
download_models.py — Pre-download all local model weights
==========================================================
Run this ONCE before starting the app so the first real query is instant.

  cd backend
  python scripts/download_models.py

Models downloaded:
  1. CLIP ViT-B/32         ~600 MB   (HuggingFace cache)
  2. Cross-encoder reranker ~90 MB   (HuggingFace cache)
  3. Whisper small         ~242 MB   (~/.cache/whisper)

Total: ~930 MB (one-time download, then cached permanently)

For local LLM (optional — only if you want LLM_PROVIDER=ollama):
  Install Ollama from https://ollama.ai
  Then run: ollama pull phi3
"""

import sys
import time

# ── ANSI colours (Windows-safe) ────────────────────────────────────────────
def _green(s): return f"\033[92m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _cyan(s): return f"\033[96m{s}\033[0m"
def _red(s): return f"\033[91m{s}\033[0m"
def _bold(s): return f"\033[1m{s}\033[0m"

def _banner(text: str):
    print(f"\n{'─' * 60}")
    print(f"  {_bold(text)}")
    print(f"{'─' * 60}")

def _ok(label: str, elapsed: float):
    print(_green(f"  ✓ {label} ({elapsed:.1f}s)"))

def _step(label: str):
    print(_cyan(f"\n  ⬇  {label}"))

# ── Enable ANSI on Windows ──────────────────────────────────────────────────
import os
os.system("")   # trick to enable ANSI escape codes on Windows cmd/PowerShell


# ════════════════════════════════════════════════════════════════════════════
# 1. CLIP ViT-B/32
# ════════════════════════════════════════════════════════════════════════════

def download_clip(model_name: str = "openai/clip-vit-base-patch32"):
    _step(f"CLIP model: {model_name}  (~600 MB)")
    t = time.time()
    try:
        from transformers import CLIPModel, CLIPProcessor
        CLIPProcessor.from_pretrained(model_name)
        CLIPModel.from_pretrained(model_name)
        _ok("CLIP downloaded and cached", time.time() - t)
    except ImportError:
        print(_red("  ✗ 'transformers' not installed. Run: pip install -r requirements.txt"))
        sys.exit(1)
    except Exception as e:
        print(_red(f"  ✗ CLIP download failed: {e}"))
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# 2. Cross-encoder reranker
# ════════════════════════════════════════════════════════════════════════════

def download_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    _step(f"Cross-encoder reranker: {model_name}  (~90 MB)")
    t = time.time()
    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder(model_name)
        _ok("Cross-encoder downloaded and cached", time.time() - t)
    except ImportError:
        print(_red("  ✗ 'sentence-transformers' not installed. Run: pip install -r requirements.txt"))
        sys.exit(1)
    except Exception as e:
        print(_red(f"  ✗ Cross-encoder download failed: {e}"))
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# 3. Whisper
# ════════════════════════════════════════════════════════════════════════════

def download_whisper(model_size: str = "small"):
    size_map = {
        "tiny": "~75 MB", "base": "~145 MB", "small": "~242 MB",
        "medium": "~769 MB", "large": "~1.5 GB",
    }
    size_label = size_map.get(model_size, "")
    _step(f"Whisper model: {model_size}  {size_label}")
    t = time.time()
    try:
        import whisper
        whisper.load_model(model_size)
        _ok(f"Whisper '{model_size}' downloaded and cached", time.time() - t)
    except ImportError:
        print(_red("  ✗ 'openai-whisper' not installed. Run: pip install -r requirements.txt"))
        sys.exit(1)
    except Exception as e:
        print(_red(f"  ✗ Whisper download failed: {e}"))
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    # Try to read model names from .env if available
    clip_model = "openai/clip-vit-base-patch32"
    whisper_model = "small"
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    try:
        from dotenv import dotenv_values
        env = dotenv_values(".env")
        clip_model = env.get("CLIP_MODEL", clip_model)
        whisper_model = env.get("LOCAL_WHISPER_MODEL", whisper_model)
        reranker_model = env.get("RERANKER_MODEL", reranker_model)
    except Exception:
        pass  # .env not found or dotenv not installed — use defaults

    _banner("Multimodal RAG — Model Pre-download")
    print(f"""
  This script downloads all local model weights to disk.
  Models are cached permanently — download happens only once.

  {_yellow('Models:')}
    • CLIP ViT-B/32              ~600 MB
    • Cross-encoder reranker      ~90 MB
    • Whisper ({whisper_model})           ~242 MB
    {'─' * 40}
    Total                        ~932 MB
    """)

    total_start = time.time()

    download_clip(clip_model)
    download_cross_encoder(reranker_model)
    download_whisper(whisper_model)

    total_elapsed = time.time() - total_start

    _banner("All models downloaded successfully!")
    print(f"""
  {_green('✓  CLIP')}            → ready
  {_green('✓  Cross-encoder')}   → ready
  {_green('✓  Whisper')}         → ready

  Total time: {total_elapsed:.1f}s

  {_yellow('Next steps:')}
  1. Copy .env.example → .env and set your OPENAI_API_KEY
  2. Create PostgreSQL database: createdb ragdb
  3. Start backend:   cd backend && uvicorn app.main:app --reload
  4. Start frontend:  cd frontend && npm run dev

  {_yellow('Optional — to use local LLM (Phi-3) instead of GPT-4o:')}
    Install Ollama:  https://ollama.ai
    Pull the model:  ollama pull phi3
    Set in .env:     LLM_PROVIDER=ollama
    """)


if __name__ == "__main__":
    main()

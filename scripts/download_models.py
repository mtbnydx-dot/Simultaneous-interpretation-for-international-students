"""
Download models for TransLive.
Usage:
    python scripts/download_models.py --asr
    python scripts/download_models.py --mt
    python scripts/download_models.py --all
    python scripts/download_models.py --mt --force
"""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.model_download import download_asr_model, download_mt_model


def download_asr():
    print(f"Downloading ASR model: {settings.asr_model_size or settings.asr_model_id}...")
    download_asr_model()
    print("ASR model downloaded and cached.")


def download_mt(force: bool = False):
    print(f"Downloading MT model: {settings.mt_model_id}...")
    model_id = download_mt_model(force=force)
    print(f"MT model ready: {model_id}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download TransLive models")
    parser.add_argument("--asr", action="store_true", help="Download ASR model")
    parser.add_argument("--mt", action="store_true", help="Download MT model (HY-MT1.5 GGUF)")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--force", action="store_true", help="Re-download MT model even if exists")
    args = parser.parse_args()

    if not any([args.asr, args.mt, args.all]):
        parser.print_help()
        return

    if args.all or args.asr:
        download_asr()
    if args.all or args.mt:
        download_mt(force=args.force)


if __name__ == "__main__":
    main()

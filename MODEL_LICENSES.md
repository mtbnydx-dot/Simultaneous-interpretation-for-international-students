# Model And Third-Party License Notes

This is a practical checklist for publishing TransLive source code. It is not legal advice.

## What This Repository Should Contain

The GitHub repository should contain source code, scripts, and documentation only. Do not commit downloaded model weights or local credentials.

Excluded from source upload:

- `models/`
- `.env`
- `.venv*` and `venv/`
- `dist/` and `build/`
- `logs/`
- downloaded GGUF, safetensors, ONNX, PyTorch checkpoint files

## Models

| Use | Default / Supported Model | License | Notes |
| --- | --- | --- | --- |
| Machine translation (default) | `tencent/Hy-MT2-1.8B-GGUF` | Apache-2.0 | Default since the Hy-MT2 release (2026-05). Same 1.8B size class as HY-MT1.5, 33 supported languages, and a permissive license. |
| Machine translation (legacy) | `tencent/HY-MT1.5-1.8B-GGUF` | Tencent HY Community License Agreement | Still supported if the file is present locally, but no longer the default. Restrictive community license: it limits the territory, requires downstream license/notice handling for redistribution, and includes acceptable-use restrictions. Do not treat it as MIT/Apache. |
| ASR on Apple Silicon (default) | `mlx-community/Qwen3-ASR-1.7B-8bit` | Apache-2.0 | Default MLX 8-bit Qwen3-ASR conversion. The app pins the verified Hugging Face snapshot revision and downloads it separately on demand. |
| Optional legacy ASR | `mlx-community/whisper-large-v3-turbo` | MIT | Source-only compatibility option; not included in the macOS Qwen-only distribution. |
| Optional legacy ASR | `openai/whisper-large-v3-turbo` | MIT | Source-only Transformers/CT2 option; not included in the macOS distribution. |
| Optional legacy ASR conversion | `dropbox-dash/faster-whisper-large-v3-turbo` | MIT | Source-only CTranslate2 option; not included in the macOS distribution. |
| Optional cloud/server ASR | `nvidia/parakeet-tdt-0.6b-v3` | CC-BY-4.0 | Candidate for NVIDIA server-side ASR, not bundled with the desktop app. |
| Optional cloud/server ASR | `nvidia/canary-1b-v2` | CC-BY-4.0 | Multilingual server-side ASR/translation candidate, not bundled with the desktop app. |
| Optional quality MT | `tencent/Hunyuan-MT-7B` | See model card/license | Candidate for cloud/high-memory quality tier; review license before use or redistribution. |
| VAD | `snakers4/silero-vad` | MIT | ONNX weights (`silero_vad.onnx`) run through onnxruntime; no PyTorch dependency. Version pinned through `TRANS_VAD_SILERO_REPO`. |

## Runtime Libraries

| Component | License | Notes |
| --- | --- | --- |
| `faster-whisper` | MIT | Optional non-Apple-Silicon source runtime; not bundled in the macOS App. |
| `CTranslate2` | MIT | Optional non-Apple-Silicon source runtime; not bundled in the macOS App. |
| `llama-cpp-python` | MIT | Python bindings used to run GGUF translation models. |
| `mlx` | MIT | Apple Silicon array and Metal runtime used by Qwen3-ASR. |
| `mlx-audio` | MIT | MLX audio runtime used to execute the default Qwen3-ASR model. |

Other Python packages are listed in `requirements.txt` and `requirements-app.txt`. If you publish binary builds, you should also review the licenses of packaged wheels and bundled dynamic libraries.

## Translation Model Publishing Checklist

The default is now Hy-MT2 (Apache-2.0), which removes the licensing constraints that HY-MT1.5 carried:

- Keep model weights out of the GitHub repository and release zip regardless of license; they are large and get re-downloaded on first run.
- Link users to the official model page before downloading.
- Apache-2.0 requires you to retain the license text and attribution notices if you redistribute the weights.

If you switch back to HY-MT1.5 (`TRANS_MT_MODEL_ID=models/HY-MT1.5-1.8B-Q4_K_M.gguf`), the stricter rules apply again:

- Review the full Tencent HY license before shipping weights to anyone else.
- Make the app/documentation clear that the translation model is Tencent HY-MT1.5 and that Tencent is not the provider or sponsor of TransLive.
- If you distribute model weights or a product that bundles/uses them for third parties, review the license's distribution, notice, territory, service disclosure, and acceptable-use clauses first.
- For commercial charging or public distribution, get proper legal review before launch.

## Sources

- Tencent Hy-MT2 GGUF: https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF
- Tencent Hy-MT2 project: https://github.com/Tencent-Hunyuan/Hy-MT2
- Tencent HY-MT1.5 GGUF: https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF
- Tencent HY-MT1.5 license: https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF/blob/main/License.txt
- Qwen3-ASR 1.7B: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- MLX Qwen3-ASR 1.7B 8-bit: https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit
- OpenAI Whisper large-v3-turbo: https://huggingface.co/openai/whisper-large-v3-turbo
- MLX Whisper large-v3-turbo 8-bit: https://huggingface.co/mlx-community/whisper-large-v3-turbo-8bit
- MLX Whisper large-v3-turbo: https://huggingface.co/mlx-community/whisper-large-v3-turbo
- CTranslate2 Whisper large-v3-turbo conversion: https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo
- Silero VAD: https://github.com/snakers4/silero-vad
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- CTranslate2: https://github.com/OpenNMT/CTranslate2
- llama-cpp-python: https://github.com/abetlen/llama-cpp-python
- MLX: https://github.com/ml-explore/mlx
- mlx-whisper: https://github.com/ml-explore/mlx-examples/tree/main/whisper
- mlx-audio: https://github.com/Blaizzy/mlx-audio
- WhisperKit / Argmax OSS Swift: https://github.com/argmaxinc/argmax-oss-swift
- NVIDIA Parakeet TDT 0.6B v3: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- NVIDIA Canary 1B v2: https://huggingface.co/nvidia/canary-1b-v2
- Tencent Hunyuan-MT-7B: https://huggingface.co/tencent/Hunyuan-MT-7B

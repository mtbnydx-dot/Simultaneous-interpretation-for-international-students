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
| Machine translation | `tencent/HY-MT1.5-1.8B-GGUF` | Tencent HY Community License Agreement | Restrictive community license. The current license limits the territory, requires downstream license/notice handling for redistribution, and includes acceptable-use restrictions. Do not treat it as MIT/Apache. |
| ASR | `openai/whisper-large-v3-turbo` | MIT | Used through Transformers or converted CTranslate2/faster-whisper formats. |
| ASR CT2 conversion | `dropbox-dash/faster-whisper-large-v3-turbo` | MIT | Converted Whisper large-v3-turbo weights for CTranslate2/faster-whisper. |
| VAD | `snakers4/silero-vad` | MIT | Loaded through torch.hub by default, pinned through `TRANS_VAD_SILERO_REPO`. |

## Runtime Libraries

| Component | License | Notes |
| --- | --- | --- |
| `faster-whisper` | MIT | ASR runtime wrapper around CTranslate2. |
| `CTranslate2` | MIT | Optimized inference runtime used by faster-whisper. |
| `llama-cpp-python` | MIT | Python bindings used to run GGUF translation models. |

Other Python packages are listed in `requirements.txt` and `requirements-app.txt`. If you publish binary builds, you should also review the licenses of packaged wheels and bundled dynamic libraries.

## HY-MT1.5 Publishing Checklist

If you keep HY-MT1.5 as the default translation model:

- Keep model weights out of the GitHub repository and release zip unless you have reviewed the full Tencent HY license.
- Link users to the official model page and license before downloading.
- Make the app/documentation clear that the translation model is Tencent HY-MT1.5 and that Tencent is not the provider or sponsor of TransLive.
- If you distribute model weights or a product that bundles/uses them for third parties, review the license's distribution, notice, territory, service disclosure, and acceptable-use clauses first.
- For commercial charging or public distribution, get proper legal review before launch.

## Sources

- Tencent HY-MT1.5 GGUF: https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF
- Tencent HY-MT1.5 license: https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF/blob/main/License.txt
- OpenAI Whisper large-v3-turbo: https://huggingface.co/openai/whisper-large-v3-turbo
- CTranslate2 Whisper large-v3-turbo conversion: https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo
- Silero VAD: https://github.com/snakers4/silero-vad
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- CTranslate2: https://github.com/OpenNMT/CTranslate2
- llama-cpp-python: https://github.com/abetlen/llama-cpp-python

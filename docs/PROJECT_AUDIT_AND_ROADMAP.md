# TransLive Compatibility Audit And Model Roadmap

Last updated: 2026-08-07

## Executive Summary

TransLive's current architecture is workable for a local desktop translator:
FastAPI/WebSocket does orchestration, the browser or desktop shell captures
audio, ASR runs locally, and Hy-MT2 GGUF handles local translation.

The main compatibility risk is not the web UI. It is runtime selection. macOS,
Windows, Intel CPU, NVIDIA CUDA, and future cloud deployments should not share a
single hard-coded model/runtime path. The project now has a hardware detection
layer, a model plan endpoint, and a generic cloud inference bridge so desktop
builds can start local-first while leaving room for paid server inference.

## What Changed In Code

- `TRANS_MODEL_PROFILE=auto` was added as the high-level policy knob.
- `TRANS_ASR_BACKEND=auto` now routes Apple Silicon to Qwen3-ASR on MLX, NVIDIA to
  faster-whisper/CTranslate2 CUDA, Intel GPU to OpenVINO when available, and CPU
  to CTranslate2 int8.
- `/api/health` now exposes `hardware`, `model_profile`, `model_plan`, and
  `cloud` fields for UI/debugging.
- ASR loading and model downloading now use the same hardware-aware recommended
  model where no explicit `TRANS_ASR_MODEL_ID` is configured.
- `TRANS_MT_BACKEND=cloud` and `TRANS_CLOUD_*` settings were added.
- Cloud ASR/MT client scaffolding was added with simple JSON and NDJSON
  streaming contracts.
- REST `/api/translate` now accepts `glossary`, matching the WebSocket path.

## Compatibility Findings

### macOS Apple Silicon

Current Python/MLX is the practical path because CTranslate2/faster-whisper does
not provide Apple MPS acceleration. The default plan uses
`mlx-community/Qwen3-ASR-1.7B-8bit` through `mlx-audio` and falls back to MLX
Whisper only when Qwen3 cannot load.

Source: https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit

For a more mature native macOS product, WhisperKit/CoreML should be evaluated as
the next major ASR runtime. It is built for Apple platforms, has documented
macOS/iOS model recommendations, and also exposes a local OpenAI-compatible
audio server route.

Source: https://github.com/argmaxinc/argmax-oss-swift

### Windows And Linux With NVIDIA

The best local/server ASR route remains CTranslate2/faster-whisper for Whisper
models when CUDA is available. On dedicated server hardware, NVIDIA NeMo/Riva/NIM
profiles should be evaluated separately from desktop builds.

Parakeet and Canary are especially relevant for server-side ASR:

- NVIDIA Parakeet TDT 0.6B v3: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- NVIDIA Canary 1B v2: https://huggingface.co/nvidia/canary-1b-v2

These are not drop-in desktop dependencies. They are better treated as cloud ASR
profiles behind the `TRANS_CLOUD_*` bridge.

### CPU-Only Machines

CPU-only real-time ASR is the weakest product tier. CTranslate2 int8 should be
used by default, with shorter audio segments and smaller models on low-memory
devices. For machines below roughly 8 GB RAM, the product should strongly
recommend cloud ASR/MT.

### Translation Models

The default local MT model is `tencent/Hy-MT2-1.8B-GGUF`. It remains in the same
small 1.8B size class, supports more languages, and uses Apache-2.0. Existing
HY-MT1.5 files remain compatible but are no longer selected first.

For quality-oriented or cloud tiers, evaluate:

- Hunyuan-MT-7B / Chimera: https://huggingface.co/tencent/Hunyuan-MT-7B
- Default Hy-MT2 GGUF: https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF
- Legacy HY-MT1.5 GGUF: https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF

Do not bundle these model weights into GitHub or app releases unless the license
and distribution terms have been reviewed.

## Runtime Routing Table

| Hardware/Profile | ASR Runtime | ASR Model | MT Runtime |
| --- | --- | --- | --- |
| Apple Silicon + MLX | `qwen3` | `mlx-community/Qwen3-ASR-1.7B-8bit` | local Hy-MT2 GGUF |
| NVIDIA CUDA | `ct2` / CUDA float16 | `large-v3-turbo` | local Hy-MT2 or cloud |
| Intel GPU + OpenVINO | `openvino` int8 | `openai/whisper-large-v3-turbo` export | local Hy-MT2 |
| CPU only | `ct2` int8 | `small` on low RAM, otherwise `large-v3-turbo` | local Hy-MT2 or cloud |
| Cloud profile | `cloud` | server selected | server selected |

## Cloud Inference Contract

The client intentionally uses simple HTTP JSON so it can sit behind a FastAPI
service, an OpenAI-compatible adapter, NVIDIA Riva/NIM, or a private gateway.

### Auth

If `TRANS_CLOUD_API_KEY` is set:

```http
Authorization: Bearer <key>
```

Extra static headers can be provided with `TRANS_CLOUD_HEADERS` as JSON.

### ASR

`POST /v1/asr/transcribe`

```json
{
  "audio_pcm16_b64": "<base64 little-endian pcm16>",
  "sample_rate": 16000,
  "language": "en",
  "model": "server-model-name",
  "provider": "generic"
}
```

Expected response:

```json
{
  "text": "recognized text",
  "language": "en",
  "duration": 3.2,
  "segments": []
}
```

### MT

`POST /v1/mt/translate`

```json
{
  "text": "hello",
  "source_lang": "en",
  "target_lang": "zh",
  "glossary": {"TransLive": "TransLive"},
  "model": "server-model-name",
  "provider": "generic",
  "stream": false
}
```

Expected response:

```json
{
  "translated": "你好"
}
```

### Streaming MT

`POST /v1/mt/translate-stream`

Response content type should be `application/x-ndjson`. Each line can contain:

```json
{"token": "你"}
{"token": "好"}
```

If the stream endpoint returns HTTP 404, the desktop client falls back to the
non-streaming MT endpoint.

## Remaining Recommendations

- Add a UI panel that shows the selected `model_plan` and warns when cloud is
  configured but unreachable.
- Add a mocked cloud server test for ASR, MT, and streaming MT contracts.
- Evaluate WhisperKit/CoreML as a native macOS runtime or sidecar local server.
- Keep Python/MLX as the practical short-term desktop path.
- Keep server-only ASR models such as Parakeet/Canary behind the cloud bridge,
  not inside the PyInstaller app.
- Split release profiles: source release, signed local app, trial app with
  embedded expiry, and notarization zip.
- Keep model weights, `.env`, virtual environments, logs, `dist/`, and `build/`
  out of GitHub.

import secrets
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from app.version import APP_CREDIT, APP_NAME, APP_VERSION

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    app_name: str = APP_NAME
    app_version: str = APP_VERSION
    app_credit: str = APP_CREDIT
    app_expiry_date: str | None = None
    host: str = "127.0.0.1"
    port: int = 8766
    # 每次进程启动都生成新的本地凭据。桌面启动器会在导入应用前通过环境变量
    # 显式设置，保证浏览器、原生音频桥和后端使用同一份值。
    local_api_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    instance_id: str = Field(default_factory=lambda: secrets.token_hex(16))

    # ── Hardware / model policy ─────────────────────────
    # auto = 根据硬件选择本地模型；speed/balanced/quality/cloud 用于产品化档位。
    model_profile: str = "auto"

    # ── ASR ──────────────────────────────────────────────
    asr_backend: str = "auto"
    asr_model_size: str = "large-v3"
    asr_device: str = "auto"
    asr_compute_type: str = "default"
    asr_max_new_tokens: int = 96
    asr_beam_size: int = 1
    asr_best_of: int = 1
    asr_cpu_threads: int = 0
    # 源码运行时可保留跨平台后端回退；macOS 冻结包会强制关闭，确保只运行 Qwen3。
    asr_fallbacks_enabled: bool = True
    # Session 已经做了分段/VAD；CT2 内置 VAD 默认关闭，避免重复切段。
    asr_ct2_vad_filter: bool = False

    # Transformers 后端配置
    asr_model_id: str | None = None
    # Qwen3-ASR（Apache-2.0，52 语言/方言），经 mlx-audio 运行。
    # mlx-community 提供 4bit/5bit/6bit/8bit/bf16 全套量化。
    # 默认 8bit：实测比 bf16 快约 30%、少占 1.6GB 内存，准确率不降反略好。
    # 也可以直接填 "4bit"/"6bit"/"bf16"/"0.6b" 这些简写。
    asr_qwen3_model_id: str = "mlx-community/Qwen3-ASR-1.7B-8bit"
    asr_mlx_fallback_model_id: str = "mlx-community/whisper-large-v3-turbo"
    hf_model_revisions: dict[str, str] = {
        "mlx-community/Qwen3-ASR-1.7B-8bit": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
        "mlx-community/whisper-large-v3-turbo": "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    }
    asr_model_max_download_gb: float = 8.0
    asr_mlx_warmup: bool = True
    asr_transformers_dtype: str = "auto"
    asr_transformers_attn: str = "sdpa"
    asr_condition_on_prev_tokens: bool = False
    asr_temperature: list[float] = [0.0]
    asr_compression_ratio_threshold: float | None = 1.35
    asr_logprob_threshold: float | None = -1.0
    asr_no_speech_threshold: float | None = 0.6

    # 实验性流式识别预览。Qwen3 的同一次正式解码会逐 token 上报草稿，
    # 不再对滚动音频重复推理。默认关闭，保持稳定模式的 UI 行为不变。
    asr_streaming_preview_enabled: bool = False
    asr_streaming_preview_min_interval_ms: int = 80
    asr_streaming_preview_min_chars: int = 2

    # Whisper 在静音、噪声或音乐尾巴上容易幻觉出 "okay" / "thank you" 等短句。
    asr_hallucination_filter_enabled: bool = True
    asr_hallucination_max_duration: float = 1.8
    asr_hallucination_low_rms: float = 0.012
    asr_hallucination_low_speech_ratio: float = 0.16
    asr_hallucination_max_speech_duration: float = 0.75
    asr_min_content_chars: int = 2
    asr_hallucination_phrases: list[str] = [
        "ok",
        "okay",
        "okey",
        "okay thank you",
        "all right",
        "alright",
        "thank you",
        "thanks",
        "thank you for watching",
        "thanks for watching",
        "subtitles by",
        "subtitles by the amara.org community",
        "music",
        "applause",
        "you",
        "bye",
        "bye bye",
        "hello",
        "yeah",
        "uh",
        "um",
        "hmm",
        "嗯",
        "嗯嗯",
        "啊",
        "呃",
        "哦",
    ]

    # ── 音频预处理 ───────────────────────────────────────
    audio_preprocess_enabled: bool = True
    audio_highpass_freq: float = 80.0

    # ── 性能监控 ─────────────────────────────────────────
    perf_log_enabled: bool = True

    auto_download_models: bool = True
    load_models_on_startup: bool = True
    desktop_mode: bool = False
    max_stream_sessions: int = 1
    websocket_max_message_bytes: int = 262144
    stream_segment_queue_max_size: int = 3
    session_drain_timeout_s: float = 20.0
    transcript_history_enabled: bool = True
    transcript_history_dir: str | None = None
    transcript_history_max_memory: int = 500
    transcript_history_retention_days: int = 30
    model_min_free_disk_gb: float = 2.0

    # ── Cloud inference bridge ──────────────────────────
    cloud_enabled: bool = False
    cloud_provider: str = "generic"
    cloud_base_url: str | None = None
    cloud_api_key: str | None = None
    cloud_timeout_s: float = 30.0
    cloud_asr_endpoint: str = "/v1/asr/transcribe"
    cloud_mt_endpoint: str = "/v1/mt/translate"
    cloud_mt_stream_endpoint: str = "/v1/mt/translate-stream"
    cloud_asr_model: str | None = None
    cloud_mt_model: str | None = None
    cloud_headers: dict[str, str] | None = None

    # ── MT (local GGUF / cloud) ─────────────────────────
    mt_backend: str = "local"  # local | cloud
    # Hy-MT2（2026-05 开源）：同为 1.8B，支持语言从 HY-MT1.5 扩到 33 种，
    # 且许可证是 Apache-2.0（HY-MT1.5 是腾讯社区协议，有地域/用途限制）。
    # 仍然兼容本地已有的 HY-MT1.5 文件，见 MTEngine._find_model_file。
    mt_model_id: str = "models/Hy-MT2-1.8B-Q4_K_M.gguf"
    mt_model_sha256: dict[str, str] = {
        "Hy-MT2-1.8B-Q4_K_M.gguf": "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699",
    }
    mt_model_max_download_gb: float = 4.0
    mt_device: str = "auto"
    mt_max_new_tokens: int = 256
    mt_num_beams: int = 1
    # 字幕场景优先可复现：贪心解码。Hy-MT2 模型卡建议 0.7/top_p 0.6，可按需调。
    mt_temperature: float = 0.0
    mt_max_input_tokens: int = 512
    mt_stream_queue_max_size: int = 128
    mt_stream_partial_min_interval_ms: int = 60
    mt_stream_partial_min_chars: int = 6
    mt_stream_timeout_s: float = 45.0
    mt_stream_max_output_chars: int = 2000
    mt_stream_stop_on_disconnect: bool = True
    mt_stream_queue_put_timeout_s: float = 2.0
    mt_output_cleanup_enabled: bool = True
    mt_full_context_max_segments: int = 220
    mt_full_context_max_chars: int = 6000
    mt_full_context_max_output_tokens: int = 1536
    # CPU 线程数；None = 自动 (cpu_count // 2)
    mt_n_threads: int | None = None
    # 上下文窗口 token 数；None = 自动 (input + output + 256 模板预留)
    mt_n_ctx: int | None = None
    # 额外停止符 —— GGUF 自带 EOS 之外的兜底停止序列。
    # 默认覆盖 ChatML / Hunyuan / Llama-3 / SentencePiece 几大家族，避免模型续写。
    # 若你确认会与 HY-MT1.5 输出冲突，可在 .env 里设 TRANS_MT_STOP_TOKENS=[] 关闭。
    mt_stop_tokens: list[str] = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "</s>",
    ]

    # 术语表 (JSON 键值对，如 {"AGI": "通用人工智能", "LLM": "大语言模型"})
    # 注意：这是启动时全局默认；客户端可通过 WS {type:"glossary"} 覆盖（仅本会话生效）
    mt_glossary: dict[str, str] | None = None

    hf_token: str | None = None

    source_lang: str = "en"
    target_lang: str = "zh"

    # ── VAD ─────────────────────────────────────────────
    vad_threshold: float = 0.5
    silence_duration_ms: int = 400
    segment_min_duration: float = 0.8
    segment_max_duration: float = 8.0
    segment_silence_duration_ms: int = 400
    session_latency_mode: str = "balanced"  # low | balanced | stable
    audio_vad_rms_threshold: float = 0.01
    sample_rate: int = 16000

    # Silero VAD 配置（ONNX Runtime，不再依赖 PyTorch）
    vad_engine: str = "silero"           # "silero" | "rms"
    # 权重按 tag 固定，避免上游更新悄悄改变本应用的行为。
    # 用 TRANS_VAD_SILERO_REPO 覆盖，格式 "owner/repo:tag"。
    vad_silero_repo: str = "snakers4/silero-vad:v6.2.1"
    # 显式指定 silero_vad.onnx 路径；留空则按 models/ → pip 包 → torch hub 缓存
    # → 从上面的 tag 下载 的顺序解析。
    vad_silero_onnx_path: str | None = None
    vad_silero_sha256: str = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
    vad_silero_threshold: float = 0.6    # 语音概率阈值
    vad_silero_min_speech_ms: int = 250  # 最短语音段

    model_config = {"env_prefix": "TRANS_", "env_file": ".env", "extra": "ignore"}


settings = Settings()

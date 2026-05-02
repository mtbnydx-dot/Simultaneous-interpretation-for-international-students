from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "TransLive"
    app_version: str = "0.88 (test)"
    app_credit: str = "薛定谔的帮你偶"
    app_expiry_date: str | None = None
    host: str = "127.0.0.1"
    port: int = 8766

    # ── ASR ──────────────────────────────────────────────
    asr_backend: str = "ct2"
    asr_model_size: str = "large-v3"
    asr_device: str = "auto"
    asr_compute_type: str = "int8"
    asr_max_new_tokens: int = 256
    asr_beam_size: int = 1
    asr_best_of: int = 1
    asr_cpu_threads: int = 0
    # Session 已经做了分段/VAD；CT2 内置 VAD 默认关闭，避免重复切段。
    asr_ct2_vad_filter: bool = False

    # Transformers 后端配置
    asr_model_id: str | None = "large-v3-turbo"
    asr_transformers_dtype: str = "auto"
    asr_transformers_attn: str = "sdpa"
    asr_condition_on_prev_tokens: bool = False
    asr_temperature: list[float] = [0.0, 0.2, 0.4, 0.6]
    asr_compression_ratio_threshold: float | None = 1.35
    asr_logprob_threshold: float | None = -1.0
    asr_no_speech_threshold: float | None = 0.6

    # Whisper 在静音、噪声或音乐尾巴上容易幻觉出 "okay" / "thank you" 等短句。
    asr_hallucination_filter_enabled: bool = True
    asr_hallucination_max_duration: float = 1.8
    asr_hallucination_low_rms: float = 0.012
    asr_min_content_chars: int = 2
    asr_hallucination_phrases: list[str] = [
        "ok",
        "okay",
        "okey",
        "all right",
        "alright",
        "thank you",
        "thanks",
        "thank you for watching",
        "thanks for watching",
        "you",
        "bye",
        "bye bye",
        "hello",
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

    # ── MT (HY-MT1.5 GGUF) ──────────────────────────────
    mt_model_id: str = "models/HY-MT1.5-1.8B-Q4_K_M.gguf"
    mt_device: str = "auto"
    mt_max_new_tokens: int = 256
    mt_num_beams: int = 1
    mt_max_input_tokens: int = 512
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
    audio_vad_rms_threshold: float = 0.01
    sample_rate: int = 16000

    # Silero VAD 配置
    vad_engine: str = "silero"           # "silero" | "rms"
    # Pin the torch.hub ref so Silero updates do not change executable code
    # under this app unexpectedly. Use TRANS_VAD_SILERO_REPO to override.
    vad_silero_repo: str = "snakers4/silero-vad:v6.2.1"
    vad_silero_threshold: float = 0.6    # 语音概率阈值
    vad_silero_min_speech_ms: int = 250  # 最短语音段

    model_config = {"env_prefix": "TRANS_", "env_file": ".env", "extra": "ignore"}


settings = Settings()

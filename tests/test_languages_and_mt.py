import sys
from types import SimpleNamespace

from app.core.languages import normalize_language_code, supported_language_options
from app.mt.engine import MTEngine


def test_supported_languages_are_exposed():
    options = supported_language_options()
    codes = {item["code"] for item in options}
    assert len(options) == 19
    assert {"en", "zh", "ja", "ko", "ru", "ar", "pt"}.issubset(codes)


def test_language_normalization_keeps_direction_codes():
    assert normalize_language_code("en-US", "zh") == "en"
    assert normalize_language_code("zh_CN", "en") == "zh"
    assert normalize_language_code("not-a-language", "ja") == "ja"


def _engine_for(model_filename: str) -> MTEngine:
    """_model_generation() 按已加载的模型文件名分发 prompt 模板。"""
    engine = MTEngine()
    engine._model_path = f"/models/{model_filename}"
    return engine


def test_mt_prompt_uses_hy_mt2_template_by_default():
    # Hy-MT2 官方模板只点明目标语言，不提源语言
    engine = _engine_for("Hy-MT2-1.8B-Q4_K_M.gguf")
    assert engine._model_generation() == "v2"

    prompt = engine._build_prompt("hello", "en", "ja")
    assert "Translate the following text into Japanese" in prompt
    assert "only output the translated result" in prompt
    assert "Chinese" not in prompt
    assert prompt.endswith("hello")


def test_mt_prompt_uses_chinese_template_when_chinese_is_involved():
    engine = _engine_for("Hy-MT2-1.8B-Q4_K_M.gguf")
    prompt = engine._build_prompt("你好", "zh", "en")
    assert "翻译成英语" in prompt
    assert "只输出翻译结果" in prompt
    assert prompt.endswith("你好")


def test_mt_prompt_falls_back_to_legacy_template_for_hy_mt15():
    # 本地还留着 HY-MT1.5 的用户不能被换成 v2 模板
    engine = _engine_for("HY-MT1.5-1.8B-Q4_K_M.gguf")
    assert engine._model_generation() == "v1"

    assert "English segment into Japanese" in engine._build_prompt("hello", "en", "ja")

    chinese = engine._build_prompt("你好", "zh", "en")
    assert "中文文本翻译为英语" in chinese
    assert "不要保留原文" in chinese


def test_mt_prompt_includes_glossary_terms():
    engine = _engine_for("Hy-MT2-1.8B-Q4_K_M.gguf")

    english = engine._build_prompt("AGI is close.", "en", "ja", glossary={"AGI": "汎用人工知能"})
    assert "AGI -> 汎用人工知能" in english

    chinese = engine._build_prompt("AGI is close.", "en", "zh", glossary={"AGI": "通用人工智能"})
    assert "AGI 翻译成 通用人工智能" in chinese


def test_transcript_prompt_uses_full_context_and_line_ids():
    prompt = MTEngine()._build_transcript_prompt(
        [
            {"index": 1, "text": "This is Apple."},
            {"index": 2, "text": "It released a new chip."},
        ],
        "en",
        "ja",
    )

    assert "complete transcript" in prompt
    assert "entire transcript as context" in prompt
    assert "[1] This is Apple." in prompt
    assert "[2] It released a new chip." in prompt


def test_transcript_output_parser_maps_numbered_lines():
    translations, warning = MTEngine()._parse_transcript_output(
        "[1] 这是苹果。\n[2] 它发布了一款新芯片。",
        expected_count=2,
    )

    assert warning is None
    assert translations == ["这是苹果。", "它发布了一款新芯片。"]


def test_document_translation_uses_one_continuous_prompt(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.prompt = ""
            self.max_tokens = 0

        def tokenize(self, payload, **_kwargs):
            return list(range(max(1, len(payload) // 4)))

        def create_completion(self, *, prompt, max_tokens, **_kwargs):
            self.prompt = prompt
            self.max_tokens = max_tokens
            return {"choices": [{"text": "这是苹果。它发布了一款新芯片。"}]}

    engine = _engine_for("Hy-MT2-1.8B-Q4_K_M.gguf")
    fake_model = FakeModel()
    engine._model = fake_model
    engine._n_ctx = 8192
    monkeypatch.setattr("app.mt.engine.settings.mt_backend", "local")

    source = "This is Apple.\nIt released a new chip."
    result = engine.translate_document(source, "en", "zh")

    assert fake_model.prompt.endswith(source)
    assert "[1]" not in fake_model.prompt
    assert fake_model.max_tokens < 1536
    assert result["full_original_text"] == source
    assert result["full_translated_text"] == "这是苹果。它发布了一款新芯片。"
    assert result["warning"] is None


def test_lazy_mt_load_configures_device_before_creating_model(monkeypatch):
    created = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            created.update(kwargs)

    engine = MTEngine()
    monkeypatch.setattr("app.mt.engine.settings.mt_backend", "local")
    monkeypatch.setattr("app.mt.engine._detect_llama_device", lambda: ("metal", -1, {}))
    monkeypatch.setattr(engine, "_find_model_file", lambda: "/tmp/fake-model.gguf")
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))

    engine._ensure_loaded()

    assert engine.device == "metal"
    assert created["n_gpu_layers"] == -1

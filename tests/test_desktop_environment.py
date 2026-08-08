import os

import desktop_launcher


def test_packaged_apple_silicon_forces_qwen_only(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop_launcher, "_is_packaged_apple_silicon", lambda: True)
    monkeypatch.setattr(desktop_launcher, "_app_support_dir", lambda: tmp_path)
    for name in (
        "TRANS_ASR_BACKEND",
        "TRANS_ASR_DEVICE",
        "TRANS_ASR_COMPUTE_TYPE",
        "TRANS_ASR_FALLBACKS_ENABLED",
    ):
        monkeypatch.setenv(name, "user-override")

    desktop_launcher._prime_environment(9876, "token", "instance")

    assert os.environ["TRANS_ASR_BACKEND"] == "qwen3"
    assert os.environ["TRANS_ASR_DEVICE"] == "mlx"
    assert os.environ["TRANS_ASR_COMPUTE_TYPE"] == "8bit"
    assert os.environ["TRANS_ASR_FALLBACKS_ENABLED"] == "false"

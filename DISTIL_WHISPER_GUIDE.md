# TransLive Distil-Whisper 集成指南

## 已完成的改动

### 1. ASR 后端重构
- 创建了抽象基类 `ASRBackend`，支持多后端扩展
- 抽取了 `ct2_backend.py` (faster-whisper)
- 抽取了 `openvino_backend.py` (Intel GPU)
- 新增了 `transformers_backend.py` (Distil-Whisper / Whisper)

### 2. 新增功能模块
- **音频预处理** (`audio_preprocess.py`): 高通滤波 + AGC 自动增益控制
- **性能监控** (`perf_monitor.py`): ASR/MT 耗时日志，RTF 计算

### 3. 配置扩展
新增配置项:
- `asr_model_id`: 完整 HuggingFace model ID
- `asr_transformers_dtype`: Transformers 模型精度
- `audio_preprocess_enabled`: 音频预处理开关
- `audio_highpass_freq`: 高通滤波截止频率
- `perf_log_enabled`: 性能监控开关

### 4. 多平台支持
| 平台 | 推荐后端 | 设备 | 模型推荐 |
|------|----------|------|---------|
| Windows + NVIDIA | transformers-distil | cuda | distil-large-v3 |
| Windows + Intel iGPU | openvino | intel_gpu | whisper-medium |
| Mac Apple Silicon | ct2 | cpu/int8 | large-v3-turbo |
| CPU | ct2 | cpu | whisper-medium |

## 快速开始

### 1. 安装依赖

```bash
# 核心依赖（已安装）
pip install -r requirements.txt

# 实时桌面默认 ASR 后端
pip install faster-whisper ctranslate2

# 如需切到 transformers-whisper 高精度后端，再安装 transformers 和 accelerate
pip install transformers accelerate

# 安装 PyTorch（根据平台选择）
# NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Mac (MPS):
pip install torch torchvision torchaudio

# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. 配置 .env 文件

创建或编辑 `.env` 文件:

```bash
# 方案 1: 实时多语言（推荐桌面版）
TRANS_ASR_BACKEND=ct2
TRANS_ASR_MODEL_ID=large-v3-turbo
TRANS_ASR_DEVICE=cpu
TRANS_ASR_COMPUTE_TYPE=int8

# 方案 2: 更高精度，但更慢、更吃内存
TRANS_ASR_BACKEND=transformers-whisper
TRANS_ASR_MODEL_ID=openai/whisper-large-v3-turbo

# 方案 3: 最高精度，但更慢
TRANS_ASR_BACKEND=transformers-whisper
TRANS_ASR_MODEL_ID=openai/whisper-large-v3

# 方案 4: Intel GPU 用户（保持现有配置）
TRANS_ASR_BACKEND=openvino
TRANS_ASR_MODEL_SIZE=medium
TRANS_ASR_DEVICE=intel_gpu
```

### 3. 启动服务器

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8766
```

### 4. 测试功能

```bash
# 运行测试脚本
python test_new_features.py

# 健康检查
curl http://localhost:8766/api/health
```

## 性能对比

| 模型 | 英文 WER | 速度 (RTF) | 模型大小 | 推荐场景 |
|------|----------|------------|----------|---------|
| whisper-medium | ~8% | 0.3-0.5 | 769MB | 通用 |
| whisper-large-v3 | ~5% | 0.5-0.8 | 1.5GB | 高精度 |
| distil-medium.en | ~6% | 0.05-0.1 | 1.5GB | 英文实时 |
| whisper-large-v3-turbo + CT2 int8 | ~7-8% | 快 | 量化运行 | 多语言实时 |
| distil-large-v3 | ~5% | 0.1-0.2 | 3GB | 英文实时 |

## 架构说明

```
┌─────────────────────────────────────────────────────┐
│                   TransLive 架构                      │
│                                                       │
│  WebSocket Audio ──▶ [AudioPreprocessor] ──▶ Buffer   │
│                                                  │     │
│                              ┌───────────────────┘     │
│                              ▼                         │
│                    ┌─── ASR Engine ───┐                │
│                    │                  │                │
│              ┌─────┴─────┐   ┌───────┴────────┐       │
│              │  ct2       │   │ transformers    │       │
│              │(faster-w)  │   │ (Distil-W / W) │       │
│              └─────┬─────┘   └───────┬────────┘       │
│                    │                 │                 │
│              ┌─────┴─────────────────┴──────┐          │
│              │  openvino (legacy, Intel GPU) │          │
│              └──────────────┬───────────────┘          │
│                             ▼                          │
│                    [PerfMonitor] ──▶ MT Engine          │
│                                                       │
└─────────────────────────────────────────────────────┘
```

## 文件清单

### 新增文件
- `app/asr/backends/__init__.py` - 后端注册表
- `app/asr/backends/base.py` - ASR 后端抽象基类
- `app/asr/backends/ct2_backend.py` - faster-whisper 后端
- `app/asr/backends/openvino_backend.py` - OpenVINO 后端
- `app/asr/backends/transformers_backend.py` - Distil-Whisper 后端
- `app/core/audio_preprocess.py` - 音频预处理模块
- `app/core/perf_monitor.py` - 性能监控模块
- `test_new_features.py` - 测试脚本

### 修改文件
- `app/core/config.py` - 新增配置字段
- `app/asr/engine.py` - 重构为委托模式
- `app/core/session.py` - 集成预处理和监控
- `app/main.py` - health 端点扩展
- `requirements.txt` - 新增依赖
- `.env.example` - 新增配置示例
- `app/core/model_download.py` - 支持新后端下载

## 故障排除

### 1. transformers 未安装
```
RuntimeError: Transformers ASR model not loaded
```
**解决**: `pip install transformers accelerate`

### 2. PyTorch 未安装
```
ImportError: No module named 'torch'
```
**解决**: 根据平台安装 PyTorch（见上方安装说明）

### 3. MPS 首次推理慢
只有手动切到 `transformers-whisper` 后端时才会走 PyTorch MPS。MPS 第一次推理需要 JIT 编译（可能 10-30 秒），后续推理正常。默认 `ct2 + int8` 路径不会触发这段开销。

### 4. 模型下载失败
```
RuntimeError: MT model download failed
```
**解决**: 检查网络连接，或设置 `TRANS_HF_TOKEN` 环境变量

## 下一步

1. **测试 Whisper Turbo**: 安装依赖后，配置 `.env` 使用 `ct2 + large-v3-turbo + int8`
2. **性能调优**: 根据实际使用情况调整 VAD 参数
3. **Mac 测试**: 在 Mac 上测试 CT2 int8 实时性能
4. **生产部署**: 考虑使用 GPU 加速和模型量化

## 参考资料

- [Distil-Whisper 论文](https://arxiv.org/abs/2311.00430)
- [HuggingFace Distil-Whisper](https://huggingface.co/distil-whisper)
- [faster-whisper 文档](https://github.com/SYSTRAN/faster-whisper)
- [OpenVINO 文档](https://docs.openvino.ai/)

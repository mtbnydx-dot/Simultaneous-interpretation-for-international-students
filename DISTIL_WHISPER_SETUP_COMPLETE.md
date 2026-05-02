# Whisper ASR 配置说明

## 已完成的配置

### 1. 依赖安装
- transformers 5.6.0
- accelerate 1.13.0
- torch 2.11.0 (CPU 版本)

### 2. 配置文件
`.env` 文件已配置：
```bash
TRANS_ASR_BACKEND=ct2
TRANS_ASR_MODEL_ID=large-v3-turbo
TRANS_ASR_DEVICE=cpu
TRANS_ASR_COMPUTE_TYPE=int8
```

### 3. 测试结果
- 模型加载: ✓ 成功
- 转录功能: ✓ 正常工作
- 设备: CPU (Intel GPU 不支持 Distil-Whisper)

## 性能说明

### 当前环境 (Windows + Intel 核显)
- **后端**: transformers-distil
- **设备**: CPU
- **转录速度**: ~2 秒/秒音频 (RTF ≈ 2.0)
- **说明**: 第一次推理较慢，后续会快一些

### Mac 环境 (Apple Silicon)
- **后端**: ct2 / faster-whisper
- **设备**: CPU int8
- **推荐模型**: large-v3-turbo
- **说明**: 默认优先实时性；如需更高精度可切到 transformers-whisper + MPS

## 启动服务器

```bash
cd D:/code/TRANS
python -m uvicorn app.main:app --host 0.0.0.0 --port 8766
```

然后在浏览器中打开 http://localhost:8766

## 在 Mac 上使用

### 1. 安装依赖
```bash
pip install faster-whisper ctranslate2
```

### 2. 配置 .env
```bash
TRANS_ASR_BACKEND=ct2
TRANS_ASR_MODEL_ID=large-v3-turbo
TRANS_ASR_DEVICE=cpu
TRANS_ASR_COMPUTE_TYPE=int8
```

### 3. 启动服务器
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8766
```

## 性能对比

| 平台 | 后端 | 设备 | 预期 RTF | 说明 |
|------|------|------|----------|------|
| Windows + Intel 核显 | transformers-distil | CPU | 1.5-2.0 | 当前配置 |
| Windows + NVIDIA GPU | transformers-distil | CUDA | 0.05-0.1 | 最快 |
| Mac Apple Silicon | ct2 | CPU int8 | 视机型而定 | 实时推荐 |
| Windows + Intel 核显 | openvino | Intel GPU | 0.3-0.5 | 备选 |

## 故障排除

### 1. 模型下载慢
模型较大，首次下载需要时间。可以使用代理或等待。

### 2. 转录速度慢
- Windows: 正常，CPU 上性能有限
- Mac: 默认使用 CT2 int8 量化路径；如果手动切回 transformers-whisper，才需要确认 PyTorch MPS 可用

### 3. 内存不足
Distil-Whisper 需要约 2-3GB 内存。如果内存不足，可以：
- 使用更小的模型: `distil-whisper/distil-small.en`
- 或使用 CT2 后端: `TRANS_ASR_BACKEND=ct2`

## 下一步

1. **测试实际音频**: 使用麦克风或音频文件测试转录效果
2. **性能调优**: 根据实际情况调整 VAD 参数
3. **Mac 测试**: 在 Mac 上测试 CT2 int8 实时性能
4. **生产部署**: 考虑使用 NVIDIA GPU 以获得最佳性能

## 参考资料

- [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- [Distil-Whisper 模型](https://huggingface.co/distil-whisper/distil-medium.en)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html)

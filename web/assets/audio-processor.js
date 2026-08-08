/**
 * AudioWorklet 处理器 —— 替代已废弃的 ScriptProcessorNode。
 * 在专用音频线程中运行，不阻塞主线程。
 *
 * process() 每次只拿到一个 render quantum（128 采样点 = 8ms @16kHz）。
 * 逐个 quantum 往 WebSocket 发会产生 125 msg/s，服务端每条都要做一次
 * numpy 分配 + VAD + 缓冲扫描，纯属浪费。这里先攒够 FRAME_SAMPLES 再发。
 *
 * 1024 = 64ms = 2 个 Silero VAD 窗口（512）。选它的原因：
 *   - 消息量降到 15.6/s（1/8）
 *   - 正好是 512 的整数倍，服务端可以直接按帧切给 VAD，不产生余数
 *   - 增加的固定延迟只有 64ms，对同传无感
 */
const FRAME_SAMPLES = 1024;

class MicProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._frame = new Int16Array(FRAME_SAMPLES);
        this._filled = 0;
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || input.length === 0) return true;
        const channel = input[0];
        if (!channel) return true;

        for (let i = 0; i < channel.length; i++) {
            const s = Math.max(-1, Math.min(1, channel[i]));
            this._frame[this._filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;
            if (this._filled === FRAME_SAMPLES) {
                // slice() 复制出独立 buffer，才能安全地 transfer 出去
                const out = this._frame.slice();
                this.port.postMessage(out.buffer, [out.buffer]);
                this._filled = 0;
            }
        }
        return true;
    }
}

registerProcessor("mic-processor", MicProcessor);

/**
 * AudioWorklet 处理器 —— 替代已废弃的 ScriptProcessorNode。
 * 在专用音频线程中运行，不阻塞主线程。
 */
class MicProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (input && input.length > 0) {
            const channel = input[0];
            if (channel) {
                const int16 = new Int16Array(channel.length);
                for (let i = 0; i < channel.length; i++) {
                    const s = Math.max(-1, Math.min(1, channel[i]));
                    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
                }
                this.port.postMessage(int16.buffer, [int16.buffer]);
            }
        }
        return true;
    }
}

registerProcessor("mic-processor", MicProcessor);

(function () {
    "use strict";

    const subtitleWsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/subtitles`;
    const stage = document.getElementById("subtitleStage");
    const closeBtn = document.getElementById("closeBtn");
    const fontDownBtn = document.getElementById("fontDownBtn");
    const fontUpBtn = document.getElementById("fontUpBtn");
    const frameDownBtn = document.getElementById("frameDownBtn");
    const frameUpBtn = document.getElementById("frameUpBtn");
    const resizeGrip = document.getElementById("resizeGrip");
    const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
    const CHANNEL_NAME = "translive.subtitle";
    const MODE_KEY = "translive.subtitle.mode";
    const FONT_SCALE_KEY = "translive.subtitle.fontScale";
    const FRAME_SIZE_KEY = "translive.subtitle.frameSize";
    const SIGNAL_KEY = "translive.subtitle.signal";
    const channel = "BroadcastChannel" in window ? new BroadcastChannel(CHANNEL_NAME) : null;
    const MAX_SEGMENTS = 80;
    const DEFAULT_FRAME = { width: 960, height: 260 };
    const MIN_FRAME = { width: 420, height: 140 };
    const MAX_FRAME = { width: 1800, height: 900 };

    const segments = new Map();
    let ws = null;
    let reconnectTimer = 0;
    let mode = normalizeMode(new URLSearchParams(location.search).get("mode") || localStorage.getItem(MODE_KEY));
    let fontScale = clamp(Number(localStorage.getItem(FONT_SCALE_KEY)) || 1, 0.72, 1.8);
    let frameSize = loadFrameSize();
    let resizeRequestToken = 0;

    function normalizeMode(value) {
        return value === "translation" ? "translation" : "bilingual";
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function loadFrameSize() {
        try {
            const parsed = JSON.parse(localStorage.getItem(FRAME_SIZE_KEY) || "null");
            if (!parsed || typeof parsed !== "object") return { ...DEFAULT_FRAME };
            return normalizeFrameSize(parsed.width, parsed.height);
        } catch {
            return { ...DEFAULT_FRAME };
        }
    }

    function normalizeFrameSize(width, height) {
        return {
            width: Math.round(clamp(Number(width) || DEFAULT_FRAME.width, MIN_FRAME.width, MAX_FRAME.width)),
            height: Math.round(clamp(Number(height) || DEFAULT_FRAME.height, MIN_FRAME.height, MAX_FRAME.height)),
        };
    }

    function saveFrameSize(size) {
        frameSize = normalizeFrameSize(size.width, size.height);
        try {
            localStorage.setItem(FRAME_SIZE_KEY, JSON.stringify(frameSize));
        } catch {}
    }

    function applyFontScale() {
        document.documentElement.style.setProperty("--overlay-font-scale", String(fontScale.toFixed(2)));
        try {
            localStorage.setItem(FONT_SCALE_KEY, String(fontScale.toFixed(2)));
        } catch {}
    }

    function changeFontScale(delta) {
        fontScale = clamp(Math.round((fontScale + delta) * 100) / 100, 0.72, 1.8);
        applyFontScale();
    }

    async function getFrameSize() {
        if (window.pywebview?.api?.subtitle_window_state) {
            try {
                const state = await window.pywebview.api.subtitle_window_state();
                if (state?.ok && state.width && state.height) {
                    return normalizeFrameSize(state.width, state.height);
                }
            } catch {}
        }
        return normalizeFrameSize(window.outerWidth || frameSize.width, window.outerHeight || frameSize.height);
    }

    async function resizeFrame(width, height) {
        const next = normalizeFrameSize(width, height);
        saveFrameSize(next);
        const requestToken = ++resizeRequestToken;
        if (window.pywebview?.api?.resize_subtitle_window) {
            try {
                const result = await window.pywebview.api.resize_subtitle_window(next.width, next.height);
                if (requestToken === resizeRequestToken && result?.ok) return;
            } catch {}
        }
        try {
            window.resizeTo(next.width, next.height);
        } catch {}
    }

    async function nudgeFrame(deltaWidth, deltaHeight) {
        const current = await getFrameSize();
        await resizeFrame(current.width + deltaWidth, current.height + deltaHeight);
    }

    function escHtml(value) {
        return String(value || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function postControl(message) {
        const payload = { ...message, ts: Date.now() };
        if (channel) channel.postMessage(payload);
        try {
            localStorage.setItem(SIGNAL_KEY, JSON.stringify(payload));
        } catch {}
    }

    function setMode(nextMode, notify = true) {
        mode = normalizeMode(nextMode);
        document.body.dataset.mode = mode;
        localStorage.setItem(MODE_KEY, mode);
        modeButtons.forEach((button) => {
            button.classList.toggle("active", button.dataset.mode === mode);
        });
        render();
        if (notify) postControl({ type: "mode", mode });
    }

    function emptySegment(id) {
        return {
            segmentId: id,
            originalText: "",
            translatedText: "",
            sourceLang: "",
            targetLang: "",
            state: "pending",
            updatedAt: Date.now(),
        };
    }

    function trimSegments() {
        while (segments.size > MAX_SEGMENTS) {
            const oldest = segments.keys().next().value;
            if (oldest === undefined) break;
            segments.delete(oldest);
        }
    }

    function upsertSegmentFromSnapshot(item) {
        const id = Number(item.segment_id);
        if (!Number.isFinite(id)) return;
        segments.set(id, {
            segmentId: id,
            originalText: item.original_text || "",
            translatedText: item.translated_text || "",
            sourceLang: item.source_lang || "",
            targetLang: item.target_lang || "",
            state: item.state || "pending",
            updatedAt: (item.updated_at || Date.now() / 1000) * 1000,
        });
        trimSegments();
    }

    function upsertSegmentFromEvent(payload) {
        const id = Number(payload.segment_id);
        if (!Number.isFinite(id)) return;
        const segment = segments.get(id) || emptySegment(id);

        if (payload.source_lang) segment.sourceLang = payload.source_lang;
        if (payload.target_lang) segment.targetLang = payload.target_lang;

        if (payload.type === "original") {
            segment.originalText = payload.text || "";
            segment.state = "pending";
        } else if (payload.type === "translated_partial") {
            segment.translatedText = payload.accumulated || payload.text || "";
            segment.state = "streaming";
        } else if (payload.type === "translated") {
            segment.translatedText = payload.text || "";
            segment.state = "final";
        } else if (payload.type === "error") {
            segment.translatedText = "[Error] " + (payload.text || "未知错误");
            segment.state = "error";
        }

        segment.updatedAt = Date.now();
        segments.set(id, segment);
        trimSegments();
    }

    function applyMessage(payload) {
        if (!payload || typeof payload !== "object") return;
        if (payload.type === "snapshot") {
            segments.clear();
            (payload.segments || []).forEach(upsertSegmentFromSnapshot);
        } else if (payload.type === "clear") {
            segments.clear();
        } else if (["original", "translated_partial", "translated", "error"].includes(payload.type)) {
            upsertSegmentFromEvent(payload);
        } else {
            return;
        }
        render();
    }

    function visibleSegments() {
        const withText = [...segments.values()]
            .filter((segment) => segment.originalText || segment.translatedText)
            .sort((a, b) => a.segmentId - b.segmentId);
        return withText.slice(mode === "translation" ? -3 : -2);
    }

    function render() {
        const items = visibleSegments();
        if (!items.length) {
            stage.innerHTML = '<div class="subtitle-empty">等待字幕…</div>';
            return;
        }

        stage.innerHTML = items.map((segment) => {
            const pendingText = segment.originalText ? "正在翻译…" : "";
            const translated = segment.translatedText || pendingText;
            const classes = ["subtitle-card", segment.state === "streaming" ? "streaming" : "", segment.state === "error" ? "error" : ""]
                .filter(Boolean)
                .join(" ");
            return `
                <section class="${classes}">
                    <p class="subtitle-original">${escHtml(segment.originalText)}</p>
                    <p class="subtitle-translation">${escHtml(translated)}</p>
                </section>
            `;
        }).join("");
    }

    function connectSubtitles() {
        clearTimeout(reconnectTimer);
        ws = new WebSocket(subtitleWsUrl);

        ws.onmessage = (event) => {
            try {
                applyMessage(JSON.parse(event.data));
            } catch {}
        };

        ws.onclose = () => {
            ws = null;
            reconnectTimer = setTimeout(connectSubtitles, 1200);
        };

        ws.onerror = () => {
            try { ws.close(); } catch {}
        };
    }

    function handleControl(message) {
        if (!message || typeof message !== "object") return;
        if (message.type === "mode") {
            setMode(message.mode, false);
        } else if (message.type === "clear") {
            segments.clear();
            render();
        }
    }

    if (channel) {
        channel.onmessage = (event) => handleControl(event.data);
    }

    window.addEventListener("storage", (event) => {
        if (event.key !== SIGNAL_KEY || !event.newValue) return;
        try {
            handleControl(JSON.parse(event.newValue));
        } catch {}
    });

    modeButtons.forEach((button) => {
        button.addEventListener("click", () => setMode(button.dataset.mode));
    });

    fontDownBtn.addEventListener("click", () => changeFontScale(-0.08));
    fontUpBtn.addEventListener("click", () => changeFontScale(0.08));
    frameDownBtn.addEventListener("click", () => nudgeFrame(-120, -40));
    frameUpBtn.addEventListener("click", () => nudgeFrame(120, 40));

    resizeGrip.addEventListener("pointerdown", async (event) => {
        event.preventDefault();
        resizeGrip.setPointerCapture(event.pointerId);
        const startX = event.screenX;
        const startY = event.screenY;
        const startSize = await getFrameSize();
        let lastResizeAt = 0;

        const onPointerMove = (moveEvent) => {
            const now = Date.now();
            if (now - lastResizeAt < 32) return;
            lastResizeAt = now;
            const width = startSize.width + (moveEvent.screenX - startX);
            const height = startSize.height + (moveEvent.screenY - startY);
            resizeFrame(width, height);
        };
        const onPointerUp = (upEvent) => {
            const width = startSize.width + (upEvent.screenX - startX);
            const height = startSize.height + (upEvent.screenY - startY);
            resizeFrame(width, height);
            resizeGrip.removeEventListener("pointermove", onPointerMove);
            resizeGrip.removeEventListener("pointerup", onPointerUp);
            resizeGrip.removeEventListener("pointercancel", onPointerUp);
        };

        resizeGrip.addEventListener("pointermove", onPointerMove);
        resizeGrip.addEventListener("pointerup", onPointerUp);
        resizeGrip.addEventListener("pointercancel", onPointerUp);
    });

    closeBtn.addEventListener("click", async () => {
        if (window.pywebview?.api?.close_subtitle_window) {
            try {
                await window.pywebview.api.close_subtitle_window();
                return;
            } catch {}
        }
        window.close();
    });

    window.addEventListener("beforeunload", () => {
        clearTimeout(reconnectTimer);
        if (ws) {
            try { ws.close(); } catch {}
        }
    });

    applyFontScale();
    setMode(mode, false);
    resizeFrame(frameSize.width, frameSize.height);
    connectSubtitles();
})();

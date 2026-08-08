(function () {
    "use strict";

    const API_TOKEN_KEY = "translive.localApiToken";
    const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
    const suppliedToken = hash.get("token") || "";
    if (suppliedToken) {
        sessionStorage.setItem(API_TOKEN_KEY, suppliedToken);
        history.replaceState(null, "", `${location.pathname}${location.search}`);
    }
    const apiToken = suppliedToken || sessionStorage.getItem(API_TOKEN_KEY) || "";
    const wsProtocols = apiToken ? ["translive", `translive-auth.${apiToken}`] : ["translive"];
    const subtitleWsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/subtitles`;
    const stage = document.getElementById("subtitleStage");
    const overlayStatus = document.getElementById("overlayStatus");
    const overlayTitle = document.getElementById("overlayTitle");
    const overlayToolbar = document.querySelector(".overlay-toolbar");
    const styleReadout = document.getElementById("styleReadout");
    const closeBtn = document.getElementById("closeBtn");
    const fontDownBtn = document.getElementById("fontDownBtn");
    const fontUpBtn = document.getElementById("fontUpBtn");
    const fontPresetBtn = document.getElementById("fontPresetBtn");
    const fontWeightBtn = document.getElementById("fontWeightBtn");
    const densityBtn = document.getElementById("densityBtn");
    const bgDownBtn = document.getElementById("bgDownBtn");
    const bgUpBtn = document.getElementById("bgUpBtn");
    const frameDownBtn = document.getElementById("frameDownBtn");
    const frameUpBtn = document.getElementById("frameUpBtn");
    const resetLayoutBtn = document.getElementById("resetLayoutBtn");
    const mainWindowBtn = document.getElementById("mainWindowBtn");
    const resizeGrip = document.getElementById("resizeGrip");
    const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
    const CHANNEL_NAME = "translive.subtitle";
    const MODE_KEY = "translive.subtitle.mode";
    const FONT_SCALE_KEY = "translive.subtitle.fontScale";
    const FONT_PRESET_KEY = "translive.subtitle.fontPreset";
    const FONT_WEIGHT_KEY = "translive.subtitle.fontWeight";
    const DENSITY_KEY = "translive.subtitle.density";
    const BG_OPACITY_KEY = "translive.subtitle.bgOpacity";
    const FRAME_SIZE_KEY = "translive.subtitle.frameSize";
    const WINDOW_POS_KEY = "translive.subtitle.windowPosition";
    const SIGNAL_KEY = "translive.subtitle.signal";
    const channel = "BroadcastChannel" in window ? new BroadcastChannel(CHANNEL_NAME) : null;
    const MAX_SEGMENTS = 80;
    const DEFAULT_FRAME = { width: 960, height: 260 };
    const MIN_FRAME = { width: 420, height: 140 };
    const MAX_FRAME = { width: 1800, height: 900 };
    const FONT_PRESETS = ["system", "rounded", "serif"];
    const FONT_PRESET_LABELS = { system: "系统字体", rounded: "圆体", serif: "衬线" };
    const FONT_WEIGHTS = ["balanced", "bold", "heavy"];
    const FONT_WEIGHT_LABELS = { balanced: "标准字重", bold: "加粗", heavy: "重字重" };
    const DENSITIES = ["relaxed", "balanced", "compact"];
    const DENSITY_LABELS = { relaxed: "宽松", balanced: "标准", compact: "紧凑" };
    const TOOLBAR_HIDE_MS = 2800;

    const segments = new Map();
    let ws = null;
    let reconnectTimer = 0;
    let reconnectAttempts = 0;
    let closing = false;
    let mode = normalizeMode(new URLSearchParams(location.search).get("mode") || localStorage.getItem(MODE_KEY));
    let fontScale = clamp(Number(localStorage.getItem(FONT_SCALE_KEY)) || 1, 0.72, 1.8);
    let fontPreset = normalizeChoice(localStorage.getItem(FONT_PRESET_KEY), FONT_PRESETS, "system");
    let fontWeight = normalizeChoice(localStorage.getItem(FONT_WEIGHT_KEY), FONT_WEIGHTS, "balanced");
    let density = normalizeChoice(localStorage.getItem(DENSITY_KEY), DENSITIES, "balanced");
    let bgOpacity = clamp(Number(localStorage.getItem(BG_OPACITY_KEY)) || 0.66, 0.24, 0.9);
    let frameSize = loadFrameSize();
    let windowPosition = loadWindowPosition();
    let resizeRequestToken = 0;
    let toolbarHideTimer = 0;
    let toolbarInteracting = false;

    function desktopCall(method, ...args) {
        const fn = window.pywebview?.api?.[method];
        if (!fn) return Promise.reject(new Error("desktop_api_unavailable"));
        return fn(apiToken, ...args);
    }

    async function loadAppMetadata() {
        if (!apiToken) return;
        try {
            const response = await fetch("/api/health", {
                cache: "no-store",
                headers: { "X-TransLive-Token": apiToken },
            });
            if (!response.ok) return;
            const metadata = await response.json();
            const name = metadata.app_name || "TransLive";
            const version = metadata.app_version || "";
            const credit = metadata.app_credit || "";
            const details = [version, credit].filter(Boolean).join(" · ");
            if (overlayTitle) overlayTitle.textContent = `${name} 字幕${details ? ` · ${details}` : ""}`;
            document.title = `${name}${version ? ` ${version}` : ""} 字幕`;
        } catch {
            // 字幕 WebSocket 仍可独立工作，元数据失败不影响显示。
        }
    }

    function normalizeMode(value) {
        return value === "translation" ? "translation" : "bilingual";
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function normalizeChoice(value, allowed, fallback) {
        return allowed.includes(value) ? value : fallback;
    }

    function nextChoice(value, allowed) {
        const index = allowed.indexOf(value);
        return allowed[(index + 1 + allowed.length) % allowed.length];
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

    function loadWindowPosition() {
        try {
            const parsed = JSON.parse(localStorage.getItem(WINDOW_POS_KEY) || "null");
            if (!parsed || typeof parsed !== "object") return null;
            return normalizeWindowPosition(parsed.x, parsed.y);
        } catch {
            return null;
        }
    }

    function normalizeWindowPosition(x, y) {
        const safeX = Number(x);
        const safeY = Number(y);
        if (!Number.isFinite(safeX) || !Number.isFinite(safeY)) return null;
        return {
            x: Math.round(clamp(safeX, -5000, 10000)),
            y: Math.round(clamp(safeY, -5000, 10000)),
        };
    }

    function saveFrameSize(size) {
        frameSize = normalizeFrameSize(size.width, size.height);
        try {
            localStorage.setItem(FRAME_SIZE_KEY, JSON.stringify(frameSize));
        } catch {}
        updateStyleReadout();
    }

    function saveWindowPosition(position) {
        const normalized = normalizeWindowPosition(position?.x, position?.y);
        if (!normalized) return;
        windowPosition = normalized;
        try {
            localStorage.setItem(WINDOW_POS_KEY, JSON.stringify(normalized));
        } catch {}
    }

    function applyFontScale() {
        document.documentElement.style.setProperty("--overlay-font-scale", String(fontScale.toFixed(2)));
        try {
            localStorage.setItem(FONT_SCALE_KEY, String(fontScale.toFixed(2)));
        } catch {}
        updateStyleReadout();
    }

    function changeFontScale(delta) {
        fontScale = clamp(Math.round((fontScale + delta) * 100) / 100, 0.72, 1.8);
        applyFontScale();
    }

    function applySubtitleStyle() {
        document.body.dataset.font = fontPreset;
        document.body.dataset.weight = fontWeight;
        document.body.dataset.density = density;

        if (fontPresetBtn) {
            fontPresetBtn.title = `切换字体：${FONT_PRESET_LABELS[fontPreset]}`;
            fontPresetBtn.setAttribute("aria-label", fontPresetBtn.title);
        }
        if (fontWeightBtn) {
            fontWeightBtn.classList.toggle("active", fontWeight !== "balanced");
            fontWeightBtn.title = `切换字重：${FONT_WEIGHT_LABELS[fontWeight]}`;
            fontWeightBtn.setAttribute("aria-label", fontWeightBtn.title);
        }
        if (densityBtn) {
            densityBtn.classList.toggle("active", density !== "balanced");
            densityBtn.title = `切换字幕密度：${DENSITY_LABELS[density]}`;
            densityBtn.setAttribute("aria-label", densityBtn.title);
        }

        try {
            localStorage.setItem(FONT_PRESET_KEY, fontPreset);
            localStorage.setItem(FONT_WEIGHT_KEY, fontWeight);
            localStorage.setItem(DENSITY_KEY, density);
        } catch {}
        render();
    }

    function cycleFontPreset() {
        fontPreset = nextChoice(fontPreset, FONT_PRESETS);
        applySubtitleStyle();
    }

    function cycleFontWeight() {
        fontWeight = nextChoice(fontWeight, FONT_WEIGHTS);
        applySubtitleStyle();
    }

    function cycleDensity() {
        density = nextChoice(density, DENSITIES);
        applySubtitleStyle();
    }

    function applyBgOpacity() {
        const base = bgOpacity.toFixed(2);
        const strong = clamp(bgOpacity + 0.16, 0.36, 0.96).toFixed(2);
        document.documentElement.style.setProperty("--overlay-bg-alpha", base);
        document.documentElement.style.setProperty("--overlay-bg-strong-alpha", strong);
        try {
            localStorage.setItem(BG_OPACITY_KEY, base);
        } catch {}
        updateStyleReadout();
    }

    function updateStyleReadout() {
        if (!styleReadout) return;
        const fontPct = Math.round(fontScale * 100);
        const bgPct = Math.round(bgOpacity * 100);
        styleReadout.textContent = `${fontPct}% · ${frameSize.width}×${frameSize.height} · ${bgPct}%`;
        styleReadout.title = `字体 ${fontPct}% · 窗口 ${frameSize.width}×${frameSize.height} · 背景 ${bgPct}%`;
    }

    function changeBgOpacity(delta) {
        bgOpacity = clamp(Math.round((bgOpacity + delta) * 100) / 100, 0.24, 0.9);
        applyBgOpacity();
    }

    async function getFrameSize() {
        if (window.pywebview?.api?.subtitle_window_state) {
            try {
                const state = await desktopCall("subtitle_window_state");
                if (state?.ok && state.width && state.height) {
                    saveWindowPosition(state);
                    return normalizeFrameSize(state.width, state.height);
                }
            } catch {}
        }
        return normalizeFrameSize(window.outerWidth || frameSize.width, window.outerHeight || frameSize.height);
    }

    // 拖拽期间以前是每 32ms 发一个不 await 的 IPC 调用，多个请求在途会乱序
    // 到达，窗口就会抖。这里改成合并：只保留最新的一次目标尺寸，串行下发。
    let pendingFrameSize = null;
    let resizePumpRunning = false;

    async function pumpResize() {
        if (resizePumpRunning) return;
        resizePumpRunning = true;
        try {
            while (pendingFrameSize) {
                const next = pendingFrameSize;
                pendingFrameSize = null;
                if (window.pywebview?.api?.resize_subtitle_window) {
                    try {
                        const result = await desktopCall("resize_subtitle_window", next.width, next.height);
                        if (result?.ok) continue;
                    } catch {}
                }
                try {
                    window.resizeTo(next.width, next.height);
                } catch {}
            }
        } finally {
            resizePumpRunning = false;
        }
    }

    /** commit=true 时才写 localStorage（拖拽结束再持久化，中间过程不写盘）。 */
    function resizeFrame(width, height, commit = true) {
        const next = normalizeFrameSize(width, height);
        if (commit) {
            saveFrameSize(next);
        } else {
            frameSize = next;
            updateStyleReadout();
        }
        pendingFrameSize = next;
        resizeRequestToken++;
        const done = pumpResize();
        if (commit) done.then(pushInteractiveZones);
        return done;
    }

    async function nudgeFrame(deltaWidth, deltaHeight) {
        const current = await getFrameSize();
        await resizeFrame(current.width + deltaWidth, current.height + deltaHeight);
    }

    /**
     * 记录窗口位置/尺寸。以前是 setInterval 每 1.5 秒无条件跑一次
     * （IPC 往返 + localStorage 写入 + DOM 更新），窗口开着就一直在跑。
     * 现在只在拖拽/缩放结束、窗口隐藏和关闭前调用。
     */
    async function rememberWindowState() {
        if (!window.pywebview?.api?.subtitle_window_state) return;
        try {
            const state = await desktopCall("subtitle_window_state");
            if (!state?.ok) return;
            if (state.width && state.height) saveFrameSize(state);
            if (Number.isFinite(Number(state.x)) && Number.isFinite(Number(state.y))) {
                saveWindowPosition(state);
            }
        } catch {}
    }

    async function applySavedWindowState() {
        await resizeFrame(frameSize.width, frameSize.height);
        if (windowPosition && window.pywebview?.api?.move_subtitle_window) {
            try {
                await desktopCall("move_subtitle_window", windowPosition.x, windowPosition.y);
            } catch {}
        }
        await rememberWindowState();
    }

    async function resetOverlayLayout() {
        fontScale = 1;
        fontPreset = "system";
        fontWeight = "balanced";
        density = "balanced";
        bgOpacity = 0.66;
        windowPosition = null;
        try {
            localStorage.removeItem(WINDOW_POS_KEY);
        } catch {}
        applyFontScale();
        applySubtitleStyle();
        applyBgOpacity();
        await resizeFrame(DEFAULT_FRAME.width, DEFAULT_FRAME.height);
    }

    // segmentId -> 已创建的 DOM 节点。翻译 partial 事件每 60ms 就来一次，
    // 以前每次都重建整个 stage.innerHTML，等于让一个透明置顶窗口以 ~16Hz
    // 全量重排重绘，闪烁和耗电都是从这来的。改成按 id 复用节点、只改变化的文本。
    const cardNodes = new Map();

    function createCard(segmentId) {
        const section = document.createElement("section");
        section.className = "subtitle-card";
        section.dataset.segmentId = String(segmentId);
        const original = document.createElement("p");
        original.className = "subtitle-original";
        const translation = document.createElement("p");
        translation.className = "subtitle-translation";
        section.append(original, translation);
        return { section, original, translation };
    }

    function postControl(message) {
        const payload = { ...message, ts: Date.now() };
        if (channel) channel.postMessage(payload);
        try {
            localStorage.setItem(SIGNAL_KEY, JSON.stringify(payload));
        } catch {}
    }

    function clearToolbarHideTimer() {
        if (toolbarHideTimer) clearTimeout(toolbarHideTimer);
        toolbarHideTimer = 0;
    }

    function toolbarShouldStayVisible() {
        return toolbarInteracting ||
            overlayToolbar?.matches(":hover") ||
            overlayToolbar?.contains(document.activeElement) ||
            overlayStatus?.classList.contains("connecting") ||
            overlayStatus?.classList.contains("error");
    }

    function hideToolbar() {
        toolbarHideTimer = 0;
        if (toolbarShouldStayVisible()) {
            scheduleToolbarHide();
            return;
        }
        document.body.classList.remove("toolbar-visible");
        document.body.classList.add("toolbar-hidden");
    }

    function scheduleToolbarHide(delay = TOOLBAR_HIDE_MS) {
        clearToolbarHideTimer();
        if (toolbarShouldStayVisible()) return;
        toolbarHideTimer = setTimeout(hideToolbar, delay);
    }

    function showToolbar(options = {}) {
        document.body.classList.add("toolbar-visible");
        document.body.classList.remove("toolbar-hidden");
        measureToolbarHotZone();
        if (!options.hold) scheduleToolbarHide();
    }

    // ── 点击穿透的"不穿透"区域 ────────────────────────────────
    // 整窗 ignoresMouseEvents 会把工具栏也一起穿透掉，所以由前端上报
    // 工具栏和缩放手柄的位置，Python 侧按光标位置动态切换。
    let toolbarHotZone = 44;   // 工具栏隐藏时也保留这条热区，否则叫不回来

    function measureToolbarHotZone() {
        if (!overlayToolbar) return;
        // 只在工具栏确实可见时量，隐藏状态下它被 translateY 挪走了，量不准
        if (!document.body.classList.contains("toolbar-visible")) return;
        const rect = overlayToolbar.getBoundingClientRect();
        if (rect.height > 0) {
            const next = Math.round(rect.bottom + 6);
            if (next !== toolbarHotZone) {
                toolbarHotZone = next;
                pushInteractiveZones();
            }
        }
    }

    function pushInteractiveZones() {
        if (!window.pywebview?.api?.set_subtitle_interactive_zones) return;
        const zones = [
            { x: 0, y: 0, w: window.innerWidth, h: toolbarHotZone },
        ];
        if (resizeGrip) {
            const g = resizeGrip.getBoundingClientRect();
            if (g.width > 0 && g.height > 0) {
                zones.push({
                    x: Math.round(g.left), y: Math.round(g.top),
                    w: Math.round(g.width), h: Math.round(g.height),
                });
            }
        }
        desktopCall("set_subtitle_interactive_zones", zones).catch(() => {});
    }

    function setOverlayStatus(text, state = "connecting", title = "") {
        if (!overlayStatus) return;
        overlayStatus.textContent = text;
        overlayStatus.className = `overlay-status ${state}`;
        overlayStatus.title = title || text;
        if (state === "connected") {
            scheduleToolbarHide();
        } else {
            showToolbar({ hold: true });
        }
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

        if (payload.type === "original_preview") {
            if (segment.state !== "final" && segment.state !== "error") {
                segment.originalText = payload.text || "";
                segment.state = "preview";
            }
        } else if (payload.type === "original") {
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
        } else if (payload.type === "original_preview_clear") {
            const id = Number(payload.segment_id);
            if (segments.get(id)?.state === "preview") segments.delete(id);
        } else if (["original_preview", "original", "translated_partial", "translated", "error"].includes(payload.type)) {
            upsertSegmentFromEvent(payload);
        } else {
            return;
        }
        scheduleRender();
    }

    function visibleSegments() {
        const limits = {
            relaxed: { bilingual: 2, translation: 2 },
            balanced: { bilingual: 2, translation: 3 },
            compact: { bilingual: 3, translation: 4 },
        };
        const limit = limits[density]?.[mode] || limits.balanced[mode];
        // Map 保持插入顺序；服务端快照和实时 segment id 都是单调递增的。
        const withText = [...segments.values()]
            .filter((segment) => mode === "translation"
                ? segment.translatedText
                : (segment.originalText || segment.translatedText));
        return withText.slice(-limit);
    }

    let renderFrame = 0;
    function scheduleRender() {
        if (renderFrame) return;
        renderFrame = requestAnimationFrame(() => {
            renderFrame = 0;
            render();
        });
    }

    function renderEmpty() {
        cardNodes.clear();
        if (stage.childElementCount === 1 && stage.firstElementChild.classList.contains("subtitle-empty")) {
            return;
        }
        const placeholder = document.createElement("div");
        placeholder.className = "subtitle-empty";
        placeholder.textContent = "等待字幕…";
        stage.replaceChildren(placeholder);
    }

    function render() {
        const items = visibleSegments();
        if (!items.length) {
            renderEmpty();
            return;
        }

        const placeholder = stage.querySelector(".subtitle-empty");
        if (placeholder) placeholder.remove();

        const wanted = new Set(items.map((segment) => segment.segmentId));
        for (const [id, node] of cardNodes) {
            if (!wanted.has(id)) {
                node.section.remove();
                cardNodes.delete(id);
            }
        }

        items.forEach((segment, index) => {
            let node = cardNodes.get(segment.segmentId);
            if (!node) {
                node = createCard(segment.segmentId);
                cardNodes.set(segment.segmentId, node);
            }

            const translated = segment.translatedText ||
                (segment.originalText && segment.state !== "preview" ? "正在翻译…" : "");
            // 只在内容真的变了时写 DOM，避免无谓的重排
            if (node.original.textContent !== segment.originalText) {
                node.original.textContent = segment.originalText;
            }
            if (node.translation.textContent !== translated) {
                node.translation.textContent = translated;
            }
            node.section.classList.toggle("streaming", segment.state === "streaming");
            node.section.classList.toggle("error", segment.state === "error");
            node.section.classList.toggle("preview", segment.state === "preview");

            if (stage.children[index] !== node.section) {
                stage.insertBefore(node.section, stage.children[index] || null);
            }
        });
    }

    function connectSubtitles() {
        clearTimeout(reconnectTimer);
        if (closing) return;
        setOverlayStatus(reconnectAttempts ? "重连中" : "连接中", "connecting");
        ws = new WebSocket(subtitleWsUrl, wsProtocols);

        ws.onopen = () => {
            reconnectAttempts = 0;
            setOverlayStatus("已同步", "connected", "字幕连接正常");
        };

        ws.onmessage = (event) => {
            try {
                applyMessage(JSON.parse(event.data));
            } catch {}
        };

        ws.onclose = () => {
            ws = null;
            if (closing) return;
            const delay = Math.min(8000, 600 * Math.pow(2, reconnectAttempts));
            reconnectAttempts = Math.min(reconnectAttempts + 1, 5);
            setOverlayStatus("重连中", "connecting", `${Math.ceil(delay / 1000)} 秒后重连`);
            reconnectTimer = setTimeout(connectSubtitles, delay);
        };

        ws.onerror = () => {
            setOverlayStatus("连接错误", "error", "字幕连接错误，正在尝试恢复");
            try { ws.close(); } catch {}
        };
    }

    function handleControl(message) {
        if (!message || typeof message !== "object") return;
        if (message.type === "mode") {
            setMode(message.mode, false);
        } else if (message.type === "clear") {
            segments.clear();
            scheduleRender();
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
    fontPresetBtn.addEventListener("click", cycleFontPreset);
    fontWeightBtn.addEventListener("click", cycleFontWeight);
    densityBtn.addEventListener("click", cycleDensity);
    bgDownBtn.addEventListener("click", () => changeBgOpacity(-0.08));
    bgUpBtn.addEventListener("click", () => changeBgOpacity(0.08));
    frameDownBtn.addEventListener("click", () => nudgeFrame(-120, -40));
    frameUpBtn.addEventListener("click", () => nudgeFrame(120, 40));
    resetLayoutBtn.addEventListener("click", resetOverlayLayout);

    // 主窗口最小化后，只要还开着悬浮窗，点 Dock 图标 macOS 是不会把它恢复出来的。
    // 悬浮窗永远可见，所以这里放一个入口，保证主窗口任何时候都叫得回来。
    if (mainWindowBtn && window.pywebview?.api?.show_main_window) {
        mainWindowBtn.hidden = false;
        mainWindowBtn.addEventListener("click", () => {
            desktopCall("show_main_window").catch(() => {});
        });
    }

    resizeGrip.addEventListener("pointerdown", async (event) => {
        // 无边框窗口没有系统缩放边框，缩放只能自己做；但拖动交给系统
        // （见 easy_drag），所以这里只保留 grip 的缩放。
        event.preventDefault();
        event.stopPropagation();
        resizeGrip.setPointerCapture(event.pointerId);
        const startX = event.screenX;
        const startY = event.screenY;
        const startSize = await getFrameSize();
        let rafPending = false;
        let latest = startSize;

        const onPointerMove = (moveEvent) => {
            latest = {
                width: startSize.width + (moveEvent.screenX - startX),
                height: startSize.height + (moveEvent.screenY - startY),
            };
            // 跟随屏幕刷新，而不是固定 32ms 定时器；且拖拽过程中不写 localStorage
            if (rafPending) return;
            rafPending = true;
            requestAnimationFrame(() => {
                rafPending = false;
                resizeFrame(latest.width, latest.height, false);
            });
        };
        const onPointerUp = (upEvent) => {
            const width = startSize.width + (upEvent.screenX - startX);
            const height = startSize.height + (upEvent.screenY - startY);
            resizeFrame(width, height, true);   // 结束时才持久化
            resizeGrip.removeEventListener("pointermove", onPointerMove);
            resizeGrip.removeEventListener("pointerup", onPointerUp);
            resizeGrip.removeEventListener("pointercancel", onPointerUp);
        };

        resizeGrip.addEventListener("pointermove", onPointerMove);
        resizeGrip.addEventListener("pointerup", onPointerUp);
        resizeGrip.addEventListener("pointercancel", onPointerUp);
    });

    if (overlayToolbar) {
        overlayToolbar.addEventListener("pointerenter", () => {
            toolbarInteracting = true;
            showToolbar({ hold: true });
        });
        overlayToolbar.addEventListener("pointerleave", () => {
            toolbarInteracting = false;
            scheduleToolbarHide();
        });
        overlayToolbar.addEventListener("focusin", () => {
            toolbarInteracting = true;
            showToolbar({ hold: true });
        });
        overlayToolbar.addEventListener("focusout", () => {
            toolbarInteracting = false;
            scheduleToolbarHide();
        });
    }

    // 拖动交给 pywebview 的原生 easy_drag（Cocoa 层在 mouseDown_/mouseDragged_
    // 里直接移动 NSWindow）。
    //
    // 注意：easy_drag 一直是开着的，而原生 mouseDown_/mouseDragged_ 在 JS 之前
    // 就执行了，JS 的 preventDefault() 根本拦不住它。所以之前那套「手动 IPC 拖动」
    // 并不是替代原生拖动，而是和它同时在改同一个窗口的位置 —— 两套实现打架，
    // 这才是拖起来又卡又飘的真正原因。现在只保留原生那一套。
    //
    // 但 pywebview 的钳制算错了方向（窗口顶部越过屏幕上沿时会把窗口整个甩到
    // 屏幕外），所以拖完要让 Python 侧确认窗口还在可见区域内。
    let dragOrigin = null;
    window.addEventListener("pointerdown", (event) => {
        dragOrigin = { x: event.screenX, y: event.screenY };
    }, { passive: true });

    window.addEventListener("pointerup", (event) => {
        const origin = dragOrigin;
        dragOrigin = null;
        if (closing || !origin) return;
        // 普通点击不触发 IPC，只有真的拖动过才处理
        const moved = Math.abs(event.screenX - origin.x) + Math.abs(event.screenY - origin.y) > 4;
        if (!moved) return;
        desktopCall("ensure_subtitle_window_visible").catch(() => {});
        rememberWindowState();
    }, { passive: true });

    closeBtn.addEventListener("click", async () => {
        if (window.pywebview?.api?.close_subtitle_window) {
            try {
                await desktopCall("close_subtitle_window");
                return;
            } catch {}
        }
        window.close();
    });

    window.addEventListener("beforeunload", () => {
        closing = true;
        clearTimeout(reconnectTimer);
        clearToolbarHideTimer();
        rememberWindowState();
        if (ws) {
            try { ws.close(); } catch {}
        }
    });

    applyFontScale();
    applySubtitleStyle();
    applyBgOpacity();
    updateStyleReadout();
    setMode(mode, false);
    showToolbar();
    applySavedWindowState().then(pushInteractiveZones);
    connectSubtitles();
    loadAppMetadata();

    // 窗口尺寸变了，缩放手柄的位置和工具栏宽度都要重新上报
    window.addEventListener("resize", () => {
        measureToolbarHotZone();
        pushInteractiveZones();
    }, { passive: true });

    // 窗口被隐藏/切走时保存一次状态，替代原来每 1.5 秒的常驻轮询
    document.addEventListener("visibilitychange", () => {
        if (document.hidden && !closing) rememberWindowState();
    });

    // pointermove 会以鼠标事件频率触发（可达数百 Hz）。原来每次都
    // classList 改两下再重排一个 setTimeout，纯浪费。工具栏已经可见时直接返回。
    let lastToolbarNudge = 0;
    window.addEventListener("pointermove", () => {
        const now = Date.now();
        if (now - lastToolbarNudge < 200) return;
        lastToolbarNudge = now;
        showToolbar();
    }, { passive: true });
    window.addEventListener("keydown", () => showToolbar());
})();

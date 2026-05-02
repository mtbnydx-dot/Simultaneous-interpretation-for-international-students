(function () {
    "use strict";

    const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/stream`;
    const $ = (id) => document.getElementById(id);

    const statusEl = $("status");
    const versionPill = $("versionPill");
    const creditLine = $("creditLine");
    const asrBadge = $("asrBadge");
    const mtBadge = $("mtBadge");
    const toggleBtn = $("toggleBtn");
    const toggleLabel = toggleBtn.querySelector(".btn-label");
    const swapLangBtn = $("swapLang");
    const sourceLangSel = $("sourceLang");
    const targetLangSel = $("targetLang");
    const transcript = $("transcript");
    const emptyState = $("emptyState");
    const meterBar = $("meterBar");
    const hint = $("hint");
    const jumpPill = $("jumpPill");
    const autoScrollChk = $("autoScroll");
    const statsEl = $("stats");
    const cardTemplate = $("cardTemplate");
    const clearBtn = $("clearBtn");
    const copyBtn = $("copyBtn");
    const exportTxtBtn = $("exportTxt");
    const exportSrtBtn = $("exportSrt");
    const modelDownloadBtn = $("modelDownloadBtn");
    const glossaryToggle = $("glossaryToggle");
    const glossaryPanel = $("glossaryPanel");
    const glossaryRows = $("glossaryRows");
    const glossaryAddBtn = $("glossaryAddBtn");
    const glossaryApplyBtn = $("glossaryApplyBtn");
    const modelPrompt = $("modelPrompt");
    const modelPromptClose = $("modelPromptClose");
    const modelPromptLater = $("modelPromptLater");
    const modelPromptDownload = $("modelPromptDownload");
    const modelPromptText = $("modelPromptText");
    const modelPromptModel = $("modelPromptModel");
    const audioSourceSel = $("audioSource");
    const subtitleModeSel = $("subtitleMode");
    const subtitleWindowBtn = $("subtitleWindowBtn");

    const LANG_LABELS = {
        en: "English", zh: "中文", ja: "日本語", ko: "한국어",
        de: "Deutsch", fr: "Français", es: "Español", ru: "Русский",
        ar: "العربية", pt: "Português", it: "Italiano", th: "ไทย",
        vi: "Tiếng Việt", id: "Bahasa Indonesia", hi: "हिन्दी",
        tr: "Türkçe", pl: "Polski", cs: "Čeština", nl: "Nederlands",
    };
    const PREF_KEY = "translive.frontend.v3";
    const SUBTITLE_CHANNEL = "translive.subtitle";
    const SUBTITLE_SIGNAL_KEY = "translive.subtitle.signal";
    const subtitleChannel = "BroadcastChannel" in window ? new BroadcastChannel(SUBTITLE_CHANNEL) : null;

    let ws = null;
    let pendingConnect = null;
    let subtitleMirrorWs = null;
    let pendingSubtitleMirror = null;
    let reconnectTimer = 0;
    let reconnectAttempts = 0;
    let mediaStream = null;
    let audioContext = null;
    let audioWorkletNode = null;
    let analyser = null;
    let meterRaf = 0;
    let isRecording = false;
    let isStarting = false;
    let isDownloading = false;
    let backendReady = false;
    let backendLoading = false;
    let modelPromptDismissed = false;
    let lastHealth = null;
    let userScrolledUp = false;
    let lastSegId = 0;
    let steadyHint = hint.textContent;
    let captureMode = "mic";
    let nativeAudioAvailable = false;
    let nativeAudioInitTimer = 0;
    let nativeAudioInitAttempts = 0;
    let nativeAudioProbeRunning = false;

    const segments = new Map();

    function clearReconnect() {
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = 0;
        reconnectAttempts = 0;
    }

    function setStatus(text, cls) {
        statusEl.textContent = text;
        statusEl.className = "status" + (cls ? " " + cls : "");
    }

    function setHint(text) {
        steadyHint = text;
        hint.textContent = text;
    }

    function flashHint(msg) {
        hint.textContent = msg;
        clearTimeout(flashHint.timer);
        flashHint.timer = setTimeout(() => { hint.textContent = steadyHint; }, 1800);
    }

    function fmtTime(d = new Date()) {
        const p = (n) => String(n).padStart(2, "0");
        return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
    }

    function fmtSrtTime(ms) {
        const h = Math.floor(ms / 3600000);
        const m = Math.floor(ms / 60000) % 60;
        const s = Math.floor(ms / 1000) % 60;
        const ms3 = Math.floor(ms % 1000);
        const p = (n, w = 2) => String(n).padStart(w, "0");
        return `${p(h)}:${p(m)}:${p(s)},${p(ms3, 3)}`;
    }

    function shortModelName(modelId) {
        if (!modelId) return "";
        const parts = String(modelId).split(/[\\/]/);
        return parts[parts.length - 1] || modelId;
    }

    function langLabel(code) {
        return LANG_LABELS[code] || code || "?";
    }

    function setBadge(el, label, value, state, title) {
        const em = el.querySelector("em");
        if (el.firstChild) el.firstChild.nodeValue = `${label} `;
        if (em) em.textContent = value || "—";
        el.className = `badge${state ? " " + state : ""}`;
        el.title = title || `${label} 模型`;
    }

    function setLangPair(entry, srcLang, tgtLang) {
        if (srcLang) entry.srcLang = srcLang;
        if (tgtLang) entry.tgtLang = tgtLang;
        entry.langPair.textContent = `${langLabel(entry.srcLang)} → ${langLabel(entry.tgtLang)}`;
    }

    function optionExists(select, value) {
        return Array.from(select.options).some((option) => option.value === value);
    }

    function loadPrefs() {
        try {
            const prefs = JSON.parse(localStorage.getItem(PREF_KEY) || "{}");
            if (prefs.sourceLang && optionExists(sourceLangSel, prefs.sourceLang)) {
                sourceLangSel.value = prefs.sourceLang;
            }
            if (prefs.targetLang && optionExists(targetLangSel, prefs.targetLang)) {
                targetLangSel.value = prefs.targetLang;
            }
            if (typeof prefs.autoScroll === "boolean") {
                autoScrollChk.checked = prefs.autoScroll;
            }
            if (prefs.audioSource && optionExists(audioSourceSel, prefs.audioSource)) {
                audioSourceSel.value = prefs.audioSource;
            }
            if (prefs.subtitleMode && optionExists(subtitleModeSel, prefs.subtitleMode)) {
                subtitleModeSel.value = prefs.subtitleMode;
            }
            if (prefs.glossary && Array.isArray(prefs.glossary)) {
                restoreGlossary(prefs.glossary);
            }
        } catch {
            localStorage.removeItem(PREF_KEY);
        }
    }

    function savePrefs() {
        const glossary = getGlossaryEntries();
        localStorage.setItem(PREF_KEY, JSON.stringify({
            sourceLang: sourceLangSel.value,
            targetLang: targetLangSel.value,
            autoScroll: autoScrollChk.checked,
            audioSource: audioSourceSel.value,
            subtitleMode: subtitleModeSel.value,
            glossary: glossary,
        }));
    }

    function getGlossaryEntries() {
        const entries = [];
        glossaryRows.querySelectorAll(".glossary-row").forEach((row) => {
            const inputs = row.querySelectorAll("input");
            const src = (inputs[0]?.value || "").trim();
            const tgt = (inputs[1]?.value || "").trim();
            if (src && tgt) entries.push({ src, tgt });
        });
        return entries;
    }

    function restoreGlossary(entries) {
        entries.forEach((e) => addGlossaryRow(e.src, e.tgt));
    }

    function addGlossaryRow(src = "", tgt = "") {
        const row = document.createElement("div");
        row.className = "glossary-row";
        row.innerHTML = `
            <input type="text" value="${escHtml(src)}" placeholder="源词" class="glossary-src">
            <span class="glossary-arrow">→</span>
            <input type="text" value="${escHtml(tgt)}" placeholder="译词" class="glossary-tgt">
            <button class="glossary-del btn-icon-only" title="删除">×</button>
        `;
        row.querySelector(".glossary-del").addEventListener("click", () => {
            row.remove();
            if (glossaryRows.querySelectorAll(".glossary-row").length === 0) {
                addGlossaryRow();
            }
        });
        glossaryRows.appendChild(row);
    }

    function escHtml(s) {
        return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function sendGlossaryToServer() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({ type: "glossary", glossary: glossaryObject() }));
    }

    function glossaryObject() {
        const glossary = {};
        getGlossaryEntries().forEach((e) => { glossary[e.src] = e.tgt; });
        return glossary;
    }

    function idleHintText() {
        if (audioSourceSel.value === "system" && nativeAudioAvailable) {
            return "点击「开始同传」采集系统音频";
        }
        return "点击「开始同传」并允许麦克风权限";
    }

    function updateControls() {
        const hasSegments = segments.size > 0;
        // 录音中允许停止；非录音时必须 backendReady 才能开始
        const canToggle = isRecording ? true : backendReady;
        toggleBtn.disabled = isStarting || !canToggle;
        if (!isRecording) {
            if (backendLoading) {
                toggleLabel.textContent = "模型加载中...";
            } else if (!backendReady) {
                toggleLabel.textContent = "模型未就绪";
            } else {
                toggleLabel.textContent = "开始同传";
            }
        }
        swapLangBtn.disabled = isRecording || isStarting;
        clearBtn.disabled = !hasSegments;
        copyBtn.disabled = !hasSegments;
        exportTxtBtn.disabled = !hasSegments;
        exportSrtBtn.disabled = !hasSegments;
        modelDownloadBtn.disabled = isDownloading || isRecording || isStarting || backendLoading;
        audioSourceSel.disabled = isRecording || isStarting;
        const downloadText = isDownloading ? "下载中..." : (backendLoading ? "加载中..." : "下载模型");
        modelDownloadBtn.textContent = downloadText;
        if (modelPromptDownload) {
            modelPromptDownload.disabled = modelDownloadBtn.disabled;
            modelPromptDownload.textContent = downloadText;
        }
    }

    function looksLikeMissingModel(reason) {
        return /No HY-MT|GGUF model found|not found|download/i.test(String(reason || ""));
    }

    function showModelPrompt(health) {
        if (!modelPrompt || modelPromptDismissed || isDownloading || isRecording) return;
        const reason = health?.mt_unavailable_reason || "";
        const shouldPrompt = looksLikeMissingModel(reason) ||
            (health?.desktop_mode && !health?.ready && !health?.asr_loading && !health?.mt_loading);
        if (!shouldPrompt) return;
        const missing = [];
        if (!health?.asr_loaded) missing.push("语音识别模型");
        if (!health?.mt_loaded) missing.push("翻译模型");
        if (modelPromptModel) {
            modelPromptModel.textContent = missing.length
                ? missing.join(" / ")
                : (shortModelName(health?.mt_model_id) || "HY-MT1.5-1.8B-Q4_K_M.gguf");
        }
        if (modelPromptText) {
            modelPromptText.textContent = "安装包不内置模型。需要下载所需模型后才能开始同传。";
        }
        modelPrompt.hidden = false;
    }

    function hideModelPrompt(dismiss = false) {
        if (dismiss) modelPromptDismissed = true;
        if (modelPrompt) modelPrompt.hidden = true;
    }

    function normalizeSubtitleMode(mode) {
        return mode === "translation" ? "translation" : "bilingual";
    }

    function postSubtitleControl(message) {
        const payload = { ...message, ts: Date.now() };
        if (subtitleChannel) subtitleChannel.postMessage(payload);
        try {
            localStorage.setItem(SUBTITLE_SIGNAL_KEY, JSON.stringify(payload));
        } catch {}
    }

    function setSubtitleMode(mode, notify = true) {
        const nextMode = normalizeSubtitleMode(mode);
        if (subtitleModeSel.value !== nextMode) subtitleModeSel.value = nextMode;
        savePrefs();
        if (notify) postSubtitleControl({ type: "mode", mode: nextMode });
    }

    async function openSubtitleWindow() {
        const mode = normalizeSubtitleMode(subtitleModeSel.value);
        setSubtitleMode(mode);
        const overlayUrl = `/overlay.html?mode=${encodeURIComponent(mode)}`;
        let opened = false;

        if (window.pywebview?.api?.open_subtitle_window) {
            try {
                const result = await window.pywebview.api.open_subtitle_window(mode);
                opened = !!result?.ok;
            } catch {}
        }

        if (!opened) {
            const popup = window.open(
                overlayUrl,
                "translive_subtitles",
                "width=960,height=260,resizable=yes,menubar=no,toolbar=no,location=no,status=no",
            );
            opened = !!popup;
        }

        flashHint(opened ? "字幕窗已打开" : "字幕窗被系统拦截，请允许弹窗");
    }

    function clearSubtitleWindow() {
        postSubtitleControl({ type: "clear" });
        fetch("/api/subtitles/clear", { method: "POST" }).catch(() => {});
    }

    function applySubtitleSnapshot(payload) {
        (payload.segments || []).forEach((segment) => {
            const id = Number(segment.segment_id);
            if (!Number.isFinite(id)) return;
            if (segment.original_text) {
                applyOriginal({
                    segment_id: id,
                    text: segment.original_text,
                    audio_duration_ms: segment.audio_duration_ms,
                    asr_ms: segment.asr_ms,
                    source_lang: segment.source_lang,
                    target_lang: segment.target_lang,
                });
            }
            if (segment.translated_text) {
                applyTranslated({
                    segment_id: id,
                    text: segment.translated_text,
                    audio_duration_ms: segment.audio_duration_ms,
                    asr_ms: segment.asr_ms,
                    mt_ms: segment.mt_ms,
                    rtf: segment.rtf,
                    source_lang: segment.source_lang,
                    target_lang: segment.target_lang,
                });
            }
        });
    }

    function applySubtitleMirrorMessage(payload) {
        if (!payload || typeof payload !== "object") return;
        if (payload.type === "snapshot") applySubtitleSnapshot(payload);
        else if (payload.type === "clear") {
            segments.clear();
            transcript.innerHTML = "";
            transcript.appendChild(emptyState);
            userScrolledUp = false;
            jumpPill.hidden = true;
            updateControls();
        }
        else if (payload.type === "original") applyOriginal(payload);
        else if (payload.type === "translated_partial") applyTranslatedPartial(payload);
        else if (payload.type === "translated") applyTranslated(payload);
        else if (payload.type === "error") applyError(payload);
    }

    function connectSubtitleMirror() {
        if (subtitleMirrorWs && subtitleMirrorWs.readyState === WebSocket.OPEN) return Promise.resolve();
        if (pendingSubtitleMirror) return pendingSubtitleMirror;

        pendingSubtitleMirror = new Promise((resolve, reject) => {
            const mirrorUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/subtitles`;
            let settled = false;
            subtitleMirrorWs = new WebSocket(mirrorUrl);

            subtitleMirrorWs.onopen = () => {
                settled = true;
                resolve();
            };
            subtitleMirrorWs.onmessage = (event) => {
                try {
                    applySubtitleMirrorMessage(JSON.parse(event.data));
                } catch {}
            };
            subtitleMirrorWs.onclose = () => {
                subtitleMirrorWs = null;
                if (!settled) {
                    settled = true;
                    reject(new Error("字幕事件连接失败"));
                }
            };
            subtitleMirrorWs.onerror = () => {
                if (!settled) {
                    settled = true;
                    reject(new Error("字幕事件连接失败"));
                }
                try { subtitleMirrorWs.close(); } catch {}
            };
        }).finally(() => {
            pendingSubtitleMirror = null;
        });

        return pendingSubtitleMirror;
    }

    function closeSubtitleMirror() {
        if (subtitleMirrorWs) {
            try { subtitleMirrorWs.close(); } catch {}
            subtitleMirrorWs = null;
        }
    }

    function handleSubtitleControl(message) {
        if (!message || typeof message !== "object") return;
        if (message.type === "mode") {
            setSubtitleMode(message.mode, false);
        }
    }

    if (subtitleChannel) {
        subtitleChannel.onmessage = (event) => handleSubtitleControl(event.data);
    }

    window.addEventListener("storage", (event) => {
        if (event.key !== SUBTITLE_SIGNAL_KEY || !event.newValue) return;
        try {
            handleSubtitleControl(JSON.parse(event.newValue));
        } catch {}
    });

    function ensureCard(segId, srcLang, tgtLang) {
        const id = segId == null ? ++lastSegId : segId;
        if (segments.has(id)) {
            const existing = segments.get(id);
            setLangPair(existing, srcLang, tgtLang);
            return existing;
        }
        if (emptyState && emptyState.parentNode) emptyState.remove();

        const node = cardTemplate.content.firstElementChild.cloneNode(true);
        node.dataset.seg = id;

        const now = new Date();
        const entry = {
            card: node,
            tsEl: node.querySelector(".ts"),
            langPair: node.querySelector(".lang-pair"),
            metrics: node.querySelector(".metrics"),
            original: node.querySelector(".row.original .text"),
            translated: node.querySelector(".row.translated .text"),
            translatedRow: node.querySelector(".row.translated"),
            ts: now,
            srcLang: srcLang || sourceLangSel.value,
            tgtLang: tgtLang || targetLangSel.value,
            originalText: "",
            translatedText: "",
            asrMs: null,
            mtMs: null,
            audioMs: null,
            totalMs: null,
            rtf: null,
        };

        entry.tsEl.textContent = fmtTime(now);
        entry.tsEl.dateTime = now.toISOString();
        setLangPair(entry, entry.srcLang, entry.tgtLang);

        segments.set(id, entry);
        transcript.appendChild(node);
        updateControls();
        scrollToLatest();
        return entry;
    }

    function setMetrics(entry) {
        const parts = [];
        if (entry.audioMs != null) parts.push(`${(entry.audioMs / 1000).toFixed(1)}s`);
        if (entry.asrMs != null) parts.push(`ASR ${entry.asrMs}ms`);
        if (entry.mtMs != null) parts.push(`MT ${entry.mtMs}ms`);
        if (entry.rtf != null) parts.push(`RTF ${entry.rtf.toFixed(2)}${entry.rtf >= 1 ? " !" : ""}`);
        entry.metrics.textContent = parts.join(" · ");
    }

    function applyOriginal(payload) {
        const entry = ensureCard(
            payload.segment_id,
            payload.source_lang || sourceLangSel.value,
            payload.target_lang || targetLangSel.value,
        );
        entry.originalText = payload.text || "";
        entry.original.textContent = entry.originalText;
        if (payload.audio_duration_ms != null) entry.audioMs = payload.audio_duration_ms;
        if (payload.asr_ms != null) entry.asrMs = payload.asr_ms;
        setMetrics(entry);
        scrollToLatest();
    }

    function applyTranslatedPartial(payload) {
        const existing = segments.get(payload.segment_id);
        const entry = ensureCard(
            payload.segment_id,
            existing?.srcLang || sourceLangSel.value,
            payload.target_lang || existing?.tgtLang || targetLangSel.value,
        );
        entry.translatedText = payload.accumulated || payload.text || "";
        entry.translated.textContent = entry.translatedText;
        entry.translatedRow.classList.remove("pending", "error");
        entry.translatedRow.classList.add("streaming");
        scrollToLatest();
    }

    function applyTranslated(payload) {
        const existing = segments.get(payload.segment_id);
        const entry = ensureCard(
            payload.segment_id,
            payload.source_lang || existing?.srcLang || sourceLangSel.value,
            payload.target_lang || existing?.tgtLang || targetLangSel.value,
        );
        entry.translatedText = payload.text || "";
        entry.translated.textContent = entry.translatedText;
        entry.translatedRow.classList.remove("pending", "error", "streaming");
        if (payload.audio_duration_ms != null) entry.audioMs = payload.audio_duration_ms;
        if (payload.asr_ms != null) entry.asrMs = payload.asr_ms;
        if (payload.mt_ms != null) entry.mtMs = payload.mt_ms;
        if (payload.total_ms != null) entry.totalMs = payload.total_ms;
        if (payload.rtf != null) entry.rtf = payload.rtf;
        setMetrics(entry);
        scrollToLatest();
    }

    function applyError(payload) {
        const entry = ensureCard(payload.segment_id, sourceLangSel.value, targetLangSel.value);
        entry.translatedText = "[Error] " + (payload.text || "未知错误");
        entry.translated.textContent = entry.translatedText;
        entry.translatedRow.classList.remove("pending", "streaming");
        entry.translatedRow.classList.add("error");
        setMetrics(entry);
        scrollToLatest();
    }

    function nearBottom() {
        return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 48;
    }

    function scrollToLatest() {
        if (!autoScrollChk.checked) return;
        if (userScrolledUp && !nearBottom()) return;
        transcript.scrollTop = transcript.scrollHeight;
    }

    transcript.addEventListener("scroll", () => {
        userScrolledUp = !nearBottom();
        jumpPill.hidden = !userScrolledUp;
    });

    jumpPill.addEventListener("click", () => {
        userScrolledUp = false;
        transcript.scrollTop = transcript.scrollHeight;
        jumpPill.hidden = true;
    });

    function sessionConfig() {
        return {
            source_lang: sourceLangSel.value,
            target_lang: targetLangSel.value,
            glossary: glossaryObject(),
        };
    }

    function sendConfig() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({
            type: "config",
            source_lang: sourceLangSel.value,
            target_lang: targetLangSel.value,
        }));
        sendGlossaryToServer();
    }

    async function sendSystemAudioConfig() {
        if (captureMode !== "system" || !isRecording) return;
        if (!window.pywebview?.api?.update_system_audio_config) return;
        try {
            const result = await window.pywebview.api.update_system_audio_config(sessionConfig());
            if (!result?.ok) {
                flashHint("系统音频语言设置同步失败：" + (result?.error || "未知错误"));
            }
        } catch (err) {
            flashHint("系统音频语言设置同步失败：" + (err.message || err));
        }
    }

    function scheduleReconnect() {
        if (!backendReady || reconnectAttempts >= 5 || reconnectTimer) {
            if (reconnectAttempts >= 5) {
                setStatus("未连接");
                setHint("连接已断开，点击「开始同传」重试");
            }
            return;
        }

        const delay = Math.min(8000, 600 * Math.pow(2, reconnectAttempts));
        reconnectAttempts += 1;
        setStatus("重连中", "connecting");
        setHint(`连接断开，${Math.ceil(delay / 1000)} 秒后重连...`);

        reconnectTimer = setTimeout(async () => {
            reconnectTimer = 0;
            try {
                await connect();
                reconnectAttempts = 0;
                setHint("连接已恢复，点击「开始同传」继续");
            } catch {
                scheduleReconnect();
            }
        }, delay);
    }

    function connect() {
        if (ws && ws.readyState === WebSocket.OPEN) return Promise.resolve();
        if (pendingConnect) return pendingConnect;

        setStatus("连接中", "connecting");
        pendingConnect = new Promise((resolve, reject) => {
            let opened = false;
            let settled = false;
            ws = new WebSocket(wsUrl);
            ws.binaryType = "arraybuffer";

            ws.onopen = () => {
                opened = true;
                settled = true;
                clearReconnect();
                setStatus("已连接", "connected");
                sendConfig();
                resolve();
            };

            ws.onclose = () => {
                const wasRecording = isRecording;
                ws = null;
                if (!opened && !settled) {
                    settled = true;
                    reject(new Error("WebSocket 连接失败"));
                }
                if (wasRecording) {
                    stopRecording({
                        closeSocket: false,
                        hintText: "连接已断开，录音已停止，正在尝试重连",
                        keepReconnect: true,
                    });
                    scheduleReconnect();
                } else {
                    setStatus("未连接");
                }
            };

            ws.onerror = () => {
                if (!settled) {
                    settled = true;
                    setStatus("连接失败", "error");
                    reject(new Error("WebSocket 连接失败"));
                }
            };

            ws.onmessage = (event) => {
                let data;
                try {
                    data = JSON.parse(event.data);
                } catch {
                    return;
                }
                if (data.segment_id != null) lastSegId = Math.max(lastSegId, data.segment_id);
                if (data.type === "original") applyOriginal(data);
                else if (data.type === "translated_partial") applyTranslatedPartial(data);
                else if (data.type === "translated") applyTranslated(data);
                else if (data.type === "error") applyError(data);
            };
        }).finally(() => {
            pendingConnect = null;
        });

        return pendingConnect;
    }

    async function startRecording() {
        if (isRecording || isStarting) return;
        if (shouldUseSystemAudio()) {
            await startSystemAudioRecording();
            return;
        }
        clearReconnect();
        isStarting = true;
        updateControls();
        setStatus("准备中", "connecting");
        setHint("正在准备麦克风...");

        try {
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error("当前浏览器不支持麦克风采集");
            }

            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

            await connect();

            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            if (audioContext.state === "suspended") await audioContext.resume();

            // 使用 AudioWorklet 替代已废弃的 ScriptProcessorNode
            await audioContext.audioWorklet.addModule("/assets/audio-processor.js");
            const source = audioContext.createMediaStreamSource(mediaStream);

            // Analyser 用于电平表
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 512;
            source.connect(analyser);

            audioWorkletNode = new AudioWorkletNode(audioContext, "mic-processor");
            audioWorkletNode.port.onmessage = (event) => {
                if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
                try {
                    ws.send(event.data);
                } catch {
                    stopRecording({ hintText: "音频发送失败，已停止" });
                }
            };

            source.connect(audioWorkletNode);

            captureMode = "mic";
            isRecording = true;
            toggleLabel.textContent = "停止同传";
            toggleBtn.classList.add("active");
            setStatus("同传中", "recording");
            setHint("正在监听麦克风...");
            startMeter();
        } catch (err) {
            cleanupAudio();
            if (ws && ws.readyState !== WebSocket.CLOSED) {
                try { ws.close(); } catch {}
            }
            setStatus("未连接");
            setHint(idleHintText());
            flashHint("启动失败：" + (err.message || err));
        } finally {
            isStarting = false;
            updateControls();
        }
    }

    function shouldUseSystemAudio() {
        return audioSourceSel.value === "system" &&
            !!window.pywebview?.api?.start_system_audio_capture &&
            nativeAudioAvailable;
    }

    async function startSystemAudioRecording() {
        clearReconnect();
        isStarting = true;
        updateControls();
        setStatus("准备中", "connecting");
        setHint("正在准备系统音频采集...");

        try {
            await connectSubtitleMirror();
            await fetch("/api/subtitles/clear", { method: "POST" }).catch(() => {});
            const result = await window.pywebview.api.start_system_audio_capture(sessionConfig());
            if (!result?.ok) {
                throw new Error(result?.error || "系统音频采集启动失败");
            }

            captureMode = "system";
            isRecording = true;
            toggleLabel.textContent = "停止同传";
            toggleBtn.classList.add("active");
            meterBar.classList.add("indeterminate");
            setStatus("同传中", "recording");
            setHint("正在监听系统音频...");
        } catch (err) {
            closeSubtitleMirror();
            try { await window.pywebview?.api?.stop_system_audio_capture?.(); } catch {}
            setStatus("未连接");
            setHint(idleHintText());
            flashHint("系统音频启动失败：" + (err.message || err));
        } finally {
            isStarting = false;
            updateControls();
        }
    }

    function stopRecording(options = {}) {
        const {
            closeSocket = true,
            hintText = "已停止，点击「开始同传」继续",
            keepReconnect = false,
        } = options;
        if (!keepReconnect) clearReconnect();
        isRecording = false;
        if (captureMode === "system") {
            try { window.pywebview?.api?.stop_system_audio_capture?.(); } catch {}
            closeSubtitleMirror();
        } else {
            cleanupAudio();
        }
        captureMode = "mic";
        meterBar.classList.remove("indeterminate");

        if (closeSocket && ws) {
            try { ws.close(); } catch {}
            ws = null;
        }

        toggleLabel.textContent = "开始同传";
        toggleBtn.classList.remove("active");
        setStatus("未连接");
        setHint(hintText === "已停止，点击「开始同传」继续" ? idleHintText() : hintText);
        meterBar.style.width = "0%";
        updateControls();
    }

    function cleanupAudio() {
        stopMeter();
        if (audioWorkletNode) {
            try { audioWorkletNode.disconnect(); } catch {}
            audioWorkletNode = null;
        }
        if (analyser) {
            try { analyser.disconnect(); } catch {}
            analyser = null;
        }
        if (audioContext) {
            try { audioContext.close(); } catch {}
            audioContext = null;
        }
        meterBar.classList.remove("indeterminate");
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = null;
        }
    }

    function startMeter() {
        if (!analyser) return;
        const buf = new Uint8Array(analyser.fftSize);
        const tick = () => {
            if (!analyser) return;
            analyser.getByteTimeDomainData(buf);
            let sum = 0;
            for (let i = 0; i < buf.length; i++) {
                const v = (buf[i] - 128) / 128;
                sum += v * v;
            }
            const rms = Math.sqrt(sum / buf.length);
            const pct = Math.min(100, Math.pow(rms * 2.2, 0.72) * 100);
            meterBar.style.width = pct.toFixed(1) + "%";
            meterRaf = requestAnimationFrame(tick);
        };
        meterRaf = requestAnimationFrame(tick);
    }

    function stopMeter() {
        if (meterRaf) cancelAnimationFrame(meterRaf);
        meterRaf = 0;
    }

    async function pollHealth() {
        try {
            const response = await fetch("/api/health", { cache: "no-store" });
            if (!response.ok) return;
            const h = await response.json();
            lastHealth = h;
            if (versionPill && h.app_version) versionPill.textContent = `v${h.app_version}`;
            if (creditLine && h.app_credit) creditLine.textContent = h.app_credit;

            const asrValue = [
                h.asr_backend,
                h.asr_device,
                h.asr_compute_type,
                shortModelName(h.asr_model_id || h.asr_model_size),
            ].filter(Boolean).join(" · ");
            const mtValue = [
                shortModelName(h.mt_model_id),
                h.mt_device,
                h.mt_acceleration?.accelerated ? "GPU" :
                    (h.mt_acceleration?.status === "cpu_only" ? "CPU only" : ""),
            ].filter(Boolean).join(" · ");
            const mtAccelNote = h.mt_acceleration?.note || "";
            const mtCpuOnly = !!(h.mt_loaded && h.mt_acceleration?.status === "cpu_only");

            const asrLoading = !!h.asr_loading;
            const mtLoading = !!h.mt_loading;
            const asrState = asrLoading ? "connecting" : (h.asr_loaded ? "ok" : "warn");
            const mtState = mtLoading ? "connecting" : (h.mt_loaded ? (mtCpuOnly ? "warn" : "ok") : "warn");
            const asrText = asrLoading ? "加载中..." : (asrValue || "未加载");
            const mtText = mtLoading ? "加载中..." : (mtValue || "未加载");

            setBadge(asrBadge, "ASR", asrText, asrState, `ASR: ${asrText}`);
            setBadge(
                mtBadge, "MT", mtText, mtState,
                h.mt_unavailable_reason ? `MT: ${h.mt_unavailable_reason}` :
                    (mtAccelNote ? `MT: ${mtText}。${mtAccelNote}` : `MT: ${mtText}`),
            );

            if ((h.mt_unavailable_reason || h.desktop_mode) && !asrLoading && !mtLoading && !h.ready) {
                modelDownloadBtn.classList.add("attention");
                showModelPrompt(h);
            } else {
                modelDownloadBtn.classList.remove("attention");
                hideModelPrompt(false);
            }

            // 控制 toggle 是否可用
            const newReady = !!(h.ready ?? (h.asr_loaded && h.mt_loaded));
            const newLoading = asrLoading || mtLoading;
            if (newReady !== backendReady || newLoading !== backendLoading) {
                backendReady = newReady;
                backendLoading = newLoading;
                updateControls();
                // 在空闲提示里同步状态
                if (!isRecording && !isStarting) {
                    if (backendLoading) {
                        setHint("模型加载中，请稍候...");
                    } else if (!backendReady) {
                        setHint("模型未就绪，点击「下载模型」获取或检查后端日志");
                    } else if (mtCpuOnly) {
                        setHint("MT 正在 CPU 模式运行，可用但延迟较高");
                    } else {
                        setHint(idleHintText());
                    }
                }
            }

            const p = h.perf_stats || {};
            if (p.segments) {
                statsEl.textContent =
                    `段数 ${p.segments} · ASR ${Math.round(p.avg_asr_ms || 0)}ms` +
                    ` · MT ${Math.round(p.avg_mt_ms || 0)}ms · RTF ${(p.avg_rtf || 0).toFixed(2)}`;
            } else if (newLoading) {
                statsEl.textContent = "模型加载中...";
            } else if (mtCpuOnly) {
                statsEl.textContent = "MT CPU 模式：可用，但延迟可能较高";
            } else {
                statsEl.textContent = h.mt_unavailable_reason ? "翻译模型未就绪" : "尚无统计数据";
            }
        } catch {
            setBadge(asrBadge, "ASR", "无法读取", "warn", "无法读取后端状态");
            setBadge(mtBadge, "MT", "无法读取", "warn", "无法读取后端状态");
            statsEl.textContent = "后端状态不可用";
            if (backendReady || backendLoading) {
                backendReady = false;
                backendLoading = false;
                updateControls();
            }
        }
    }

    async function downloadModels() {
        if (isDownloading) return;
        isDownloading = true;
        hideModelPrompt(false);
        updateControls();
        setHint("正在下载或加载翻译模型...");
        let flashMessage = "";
        try {
            const response = await fetch("/api/models/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    asr: !(lastHealth?.asr_loaded ?? true),
                    mt: !(lastHealth?.mt_loaded ?? false),
                    force: false,
                }),
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            await response.json();
            await pollHealth();
            flashMessage = "模型已就绪";
            modelPromptDismissed = true;
        } catch (err) {
            flashMessage = "模型下载失败：" + (err.message || err);
            modelPromptDismissed = false;
        } finally {
            isDownloading = false;
            setHint(isRecording ? (captureMode === "system" ? "正在监听系统音频..." : "正在监听麦克风...") : idleHintText());
            updateControls();
            if (flashMessage) flashHint(flashMessage);
        }
    }

    function buildPlain() {
        const lines = [];
        const ids = [...segments.keys()].sort((a, b) => a - b);
        for (const id of ids) {
            const e = segments.get(id);
            if (e.originalText) lines.push(`[${fmtTime(e.ts)}] (${e.srcLang}) ${e.originalText}`);
            if (e.translatedText) lines.push(`[${fmtTime(e.ts)}] (${e.tgtLang}) ${e.translatedText}`);
            lines.push("");
        }
        return lines.join("\n").trim();
    }

    function buildSrt() {
        const ids = [...segments.keys()].sort((a, b) => a - b);
        if (ids.length === 0) return "";
        const t0 = segments.get(ids[0]).ts.getTime();
        const out = [];
        let idx = 1;
        for (const id of ids) {
            const e = segments.get(id);
            const text = [e.originalText, e.translatedText].filter(Boolean).join("\n");
            if (!text) continue;
            const startMs = Math.max(0, e.ts.getTime() - t0);
            const dur = e.audioMs || 3000;
            out.push(`${idx++}\n${fmtSrtTime(startMs)} --> ${fmtSrtTime(startMs + dur)}\n${text}\n`);
        }
        return out.join("\n");
    }

    function downloadBlob(text, filename, mime) {
        if (!text) {
            flashHint("当前没有可导出的内容");
            return;
        }
        const blob = new Blob([text], { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    async function copyAll() {
        const text = buildPlain();
        if (!text) {
            flashHint("当前没有可复制的内容");
            return;
        }
        try {
            await navigator.clipboard.writeText(text);
            flashHint("已复制到剪贴板");
        } catch {
            flashHint("复制失败，浏览器拒绝了剪贴板权限");
        }
    }

    function clearTranscript() {
        segments.clear();
        transcript.innerHTML = "";
        transcript.appendChild(emptyState);
        userScrolledUp = false;
        jumpPill.hidden = true;
        clearSubtitleWindow();
        updateControls();
    }

    function handleLanguageChange() {
        savePrefs();
        if (captureMode === "system" && isRecording) {
            sendSystemAudioConfig();
        } else {
            sendConfig();
        }
        if (sourceLangSel.value === targetLangSel.value) {
            flashHint("源语言和目标语言相同，将直接显示原文");
        }
    }

    // ── 术语表面板 ──────────────────────────────────────

    glossaryToggle.addEventListener("click", () => {
        const open = glossaryPanel.classList.toggle("open");
        glossaryToggle.classList.toggle("active", open);
        if (open && glossaryRows.querySelectorAll(".glossary-row").length === 0) {
            addGlossaryRow();
        }
    });

    glossaryAddBtn.addEventListener("click", () => addGlossaryRow());

    glossaryApplyBtn.addEventListener("click", () => {
        savePrefs();
        if (captureMode === "system" && isRecording) {
            sendSystemAudioConfig();
        } else {
            sendGlossaryToServer();
        }
        flashHint("术语表已应用");
    });

    // ── 事件绑定 ────────────────────────────────────────

    toggleBtn.addEventListener("click", () => {
        isRecording ? stopRecording() : startRecording();
    });

    swapLangBtn.addEventListener("click", () => {
        const src = sourceLangSel.value;
        const tgt = targetLangSel.value;
        if (!optionExists(sourceLangSel, tgt) || !optionExists(targetLangSel, src)) {
            flashHint("当前语言组合不能交换");
            return;
        }
        sourceLangSel.value = tgt;
        targetLangSel.value = src;
        handleLanguageChange();
    });

    sourceLangSel.addEventListener("change", handleLanguageChange);
    targetLangSel.addEventListener("change", handleLanguageChange);
    audioSourceSel.addEventListener("change", () => {
        savePrefs();
        setHint(idleHintText());
    });
    autoScrollChk.addEventListener("change", savePrefs);
    clearBtn.addEventListener("click", clearTranscript);
    copyBtn.addEventListener("click", copyAll);
    exportTxtBtn.addEventListener("click", () =>
        downloadBlob(buildPlain(), `translive-${Date.now()}.txt`, "text/plain;charset=utf-8"));
    exportSrtBtn.addEventListener("click", () =>
        downloadBlob(buildSrt(), `translive-${Date.now()}.srt`, "text/plain;charset=utf-8"));
    modelDownloadBtn.addEventListener("click", downloadModels);
    subtitleModeSel.addEventListener("change", () => setSubtitleMode(subtitleModeSel.value));
    subtitleWindowBtn.addEventListener("click", openSubtitleWindow);
    modelPromptDownload.addEventListener("click", downloadModels);
    modelPromptClose.addEventListener("click", () => hideModelPrompt(true));
    modelPromptLater.addEventListener("click", () => hideModelPrompt(true));

    document.addEventListener("keydown", (e) => {
        if (e.code !== "Space" || e.repeat) return;
        const t = e.target;
        if (t && ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(t.tagName)) return;
        e.preventDefault();
        toggleBtn.click();
    });

    function scheduleNativeAudioInit(delayMs = 150) {
        clearTimeout(nativeAudioInitTimer);
        nativeAudioInitTimer = setTimeout(() => {
            initNativeAudioOption();
        }, delayMs);
    }

    async function initNativeAudioOption() {
        if (nativeAudioProbeRunning) return;
        const systemOption = audioSourceSel.querySelector('option[value="system"]');
        const hasApi = !!window.pywebview?.api?.native_audio_available;
        nativeAudioAvailable = false;
        if (!hasApi) {
            if (systemOption) systemOption.disabled = true;
            audioSourceSel.title = "正在检测系统音频...";
            updateControls();
            if (nativeAudioInitAttempts < 30) {
                nativeAudioInitAttempts += 1;
                scheduleNativeAudioInit();
                return;
            }
            audioSourceSel.title = "系统音频仅支持 macOS 桌面版";
            if (audioSourceSel.value === "system") {
                audioSourceSel.value = "mic";
                savePrefs();
            }
            setHint(idleHintText());
            updateControls();
            return;
        }

        clearTimeout(nativeAudioInitTimer);
        nativeAudioInitAttempts = 0;
        nativeAudioProbeRunning = true;
        try {
            const result = await window.pywebview.api.native_audio_available();
            nativeAudioAvailable = !!result?.ok;
            audioSourceSel.title = result?.reason || "音频输入";
        } catch (err) {
            nativeAudioAvailable = false;
            audioSourceSel.title = err?.message || "系统音频检测失败";
        } finally {
            nativeAudioProbeRunning = false;
        }
        if (systemOption) systemOption.disabled = !nativeAudioAvailable;
        if (!nativeAudioAvailable && audioSourceSel.value === "system") {
            audioSourceSel.value = "mic";
            savePrefs();
        }
        setHint(idleHintText());
        updateControls();
    }

    loadPrefs();
    updateControls();
    window.addEventListener("pywebviewready", () => {
        nativeAudioInitAttempts = 0;
        scheduleNativeAudioInit(0);
    });
    initNativeAudioOption();
    pollHealth();
    setInterval(pollHealth, 5000);
})();

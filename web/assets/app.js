(function () {
    "use strict";

    const API_TOKEN_KEY = "translive.localApiToken";

    function loadApiToken() {
        const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
        const supplied = hash.get("token") || "";
        if (supplied) {
            sessionStorage.setItem(API_TOKEN_KEY, supplied);
            history.replaceState(null, "", `${location.pathname}${location.search}`);
            return supplied;
        }
        return sessionStorage.getItem(API_TOKEN_KEY) || "";
    }

    const apiToken = loadApiToken();
    const wsProtocols = apiToken ? ["translive", `translive-auth.${apiToken}`] : ["translive"];
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
    const persistHistoryChk = $("persistHistory");
    const autoUnloadChk = $("autoUnload");
    const statsEl = $("stats");
    const historyInfo = $("historyInfo");
    const cardTemplate = $("cardTemplate");
    const clearBtn = $("clearBtn");
    const copyBtn = $("copyBtn");
    const exportTxtBtn = $("exportTxt");
    const exportSrtBtn = $("exportSrt");
    const fullTextToggle = $("fullTextToggle");
    const fullTextPanel = $("fullTextPanel");
    const fullTextStatus = $("fullTextStatus");
    const fullTextBody = $("fullTextBody");
    const fullTextGenerateBtn = $("fullTextGenerateBtn");
    const fullTextCopyBtn = $("fullTextCopyBtn");
    const fullTextExportBtn = $("fullTextExportBtn");
    const fullTextCloseBtn = $("fullTextCloseBtn");
    const historyToggle = $("historyToggle");
    const historyPanel = $("historyPanel");
    const historyStatus = $("historyStatus");
    const historyList = $("historyList");
    const historySearch = $("historySearch");
    const historyStateFilter = $("historyStateFilter");
    const historyRefreshBtn = $("historyRefreshBtn");
    const historyCopyFilteredBtn = $("historyCopyFilteredBtn");
    const historyExportFilteredBtn = $("historyExportFilteredBtn");
    const historyLoadBtn = $("historyLoadBtn");
    const historyClearSavedBtn = $("historyClearSavedBtn");
    const modelDownloadBtn = $("modelDownloadBtn");
    const modelUnloadBtn = $("modelUnloadBtn");
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
    const audioSourceStatus = $("audioSourceStatus");
    const latencyModeSel = $("latencyMode");
    const streamingPreviewChk = $("streamingPreview");
    const subtitleModeSel = $("subtitleMode");
    const subtitleWindowBtn = $("subtitleWindowBtn");
    const clickThroughWrap = $("clickThroughWrap");
    const clickThroughToggle = $("clickThrough");

    const LANG_LABELS = {
        auto: "自动检测",
        en: "English", zh: "中文", ja: "日本語", ko: "한국어",
        de: "Deutsch", fr: "Français", es: "Español", ru: "Русский",
        ar: "العربية", pt: "Português", it: "Italiano", th: "ไทย",
        vi: "Tiếng Việt", id: "Bahasa Indonesia", hi: "हिन्दी",
        tr: "Türkçe", pl: "Polski", cs: "Čeština", nl: "Nederlands",
    };
    const PREF_KEY = "translive.frontend.v3";
    const HISTORY_KEY = "translive.frontend.history.v1";
    const SUBTITLE_CHANNEL = "translive.subtitle";
    const SUBTITLE_SIGNAL_KEY = "translive.subtitle.signal";
    const SUBTITLE_FRAME_SIZE_KEY = "translive.subtitle.frameSize";
    const SUBTITLE_WINDOW_POS_KEY = "translive.subtitle.windowPosition";
    const subtitleChannel = "BroadcastChannel" in window ? new BroadcastChannel(SUBTITLE_CHANNEL) : null;
    const MAX_SEGMENTS = 220;
    const HISTORY_SAVE_DEBOUNCE_MS = 350;
    const MAX_RECONNECT_ATTEMPTS = 8;
    const MAX_SUBTITLE_MIRROR_RECONNECT_ATTEMPTS = 8;
    const AUTO_UNLOAD_IDLE_MS = 10 * 60 * 1000;
    const HEALTH_POLL_ACTIVE_MS = 5000;
    const HEALTH_POLL_IDLE_MS = 12000;
    const HEALTH_POLL_HIDDEN_MS = 30000;
    const HEALTH_POLL_ERROR_MS = 15000;
    const HEALTH_FETCH_TIMEOUT_MS = 4500;

    let ws = null;
    let pendingConnect = null;
    let subtitleMirrorWs = null;
    let pendingSubtitleMirror = null;
    let reconnectTimer = 0;
    let reconnectAttempts = 0;
    let subtitleMirrorReconnectTimer = 0;
    let subtitleMirrorReconnectAttempts = 0;
    let mediaStream = null;
    let audioContext = null;
    let audioWorkletNode = null;
    let analyser = null;
    let meterRaf = 0;
    let isRecording = false;
    let isStarting = false;
    let isDownloading = false;
    let isUnloadingModels = false;
    let backendReady = false;
    let backendLoading = false;
    let modelPromptDismissed = false;
    let forceNextModelDownload = false;
    let lastHealth = null;
    let userScrolledAwayFromLatest = false;
    let lastSegId = 0;
    let steadyHint = hint.textContent;
    let captureMode = "mic";
    let nativeAudioAvailable = false;
    let nativeAudioInitTimer = 0;
    let nativeAudioInitAttempts = 0;
    let nativeAudioProbeRunning = false;
    let historySaveTimer = 0;
    let isRestoringHistory = false;
    let savedTranscriptRecords = [];
    let visibleSavedTranscriptRecords = [];
    let isHistoryLoading = false;
    let isHistoryClearing = false;
    let fullTextResult = null;
    let isFullTextTranslating = false;
    let languageOptionSignature = "";
    let autoUnloadTimer = 0;
    let lastUserActivityAt = Date.now();
    let healthPollTimer = 0;
    let pendingHealthPoll = null;
    let pendingSessionStop = null;

    const segments = new Map();

    function apiFetch(url, options = {}) {
        const headers = new Headers(options.headers || {});
        if (apiToken) headers.set("X-TransLive-Token", apiToken);
        return fetch(url, { ...options, headers });
    }

    function desktopCall(method, ...args) {
        const fn = window.pywebview?.api?.[method];
        if (!fn) return Promise.reject(new Error("desktop_api_unavailable"));
        return fn(apiToken, ...args);
    }

    function clearReconnect() {
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = 0;
        reconnectAttempts = 0;
    }

    function clearSubtitleMirrorReconnect() {
        if (subtitleMirrorReconnectTimer) clearTimeout(subtitleMirrorReconnectTimer);
        subtitleMirrorReconnectTimer = 0;
        subtitleMirrorReconnectAttempts = 0;
    }

    function setStatus(text, cls) {
        statusEl.textContent = text;
        statusEl.className = "status" + (cls ? " " + cls : "");
    }

    function setHint(text) {
        steadyHint = text;
        hint.textContent = text;
    }

    function setAudioSourceStatus(text, state = "warn", title = "") {
        if (!audioSourceStatus) return;
        audioSourceStatus.textContent = text;
        audioSourceStatus.className = `audio-source-status ${state}`;
        audioSourceStatus.title = title || text;
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

    function fmtDuration(seconds) {
        const s = Math.max(0, Number(seconds) || 0);
        if (s < 60) return `${Math.round(s)}s`;
        const m = Math.floor(s / 60);
        if (m < 60) return `${m}m`;
        const h = Math.floor(m / 60);
        const remM = m % 60;
        return remM ? `${h}h${remM}m` : `${h}h`;
    }

    function fmtBytes(bytes) {
        const n = Math.max(0, Number(bytes) || 0);
        if (n < 1024) return `${Math.round(n)}B`;
        if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
        if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)}MB`;
        return `${(n / 1024 / 1024 / 1024).toFixed(1)}GB`;
    }

    function runtimeSummary(runtime) {
        if (!runtime || !runtime.available) return "";
        const parts = [];
        if (runtime.rss_mb != null) parts.push(`内存 ${Math.round(runtime.rss_mb)}MB`);
        if (runtime.uptime_s != null) parts.push(`运行 ${fmtDuration(runtime.uptime_s)}`);
        return parts.join(" · ");
    }

    async function fetchWithTimeout(url, options = {}, timeoutMs = HEALTH_FETCH_TIMEOUT_MS) {
        const safeTimeout = Math.max(1000, timeoutMs);
        if (typeof AbortController === "undefined") {
            let timer = 0;
            const timeout = new Promise((_, reject) => {
                timer = setTimeout(() => reject(new Error("request_timeout")), safeTimeout);
            });
            try {
                return await Promise.race([apiFetch(url, options), timeout]);
            } finally {
                clearTimeout(timer);
            }
        }
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), safeTimeout);
        try {
            return await apiFetch(url, { ...options, signal: controller.signal });
        } catch (err) {
            if (err?.name === "AbortError") {
                throw new Error("request_timeout");
            }
            throw err;
        } finally {
            clearTimeout(timer);
        }
    }

    function langLabel(code) {
        return LANG_LABELS[code] || code || "?";
    }

    function syncLanguageOptions(h) {
        const options = Array.isArray(h?.supported_languages) ? h.supported_languages : [];
        if (!options.length) return;

        const normalized = options
            .map((item) => ({
                code: String(item.code || "").trim(),
                label: String(item.label || item.chinese || item.english || item.code || "").trim(),
            }))
            .filter((item) => item.code && item.label);
        if (!normalized.length) return;

        normalized.forEach((item) => {
            LANG_LABELS[item.code] = item.label;
        });

        // 源语言比目标语言多一个「自动检测」：Qwen3-ASR 自带语种识别，
        // 适合中英混说的场合。后端不支持自动检测时（比如回退到 Whisper）
        // 就不给这个选项，避免选了没效果。
        const sourceOptions = Array.isArray(h?.supported_source_languages) && h.asr_language_autodetect
            ? h.supported_source_languages
                .map((item) => ({
                    code: String(item.code || "").trim(),
                    label: String(item.label || item.chinese || item.english || item.code || "").trim(),
                }))
                .filter((item) => item.code && item.label)
            : normalized;

        const supportedCodes = new Set(normalized.map((item) => item.code));
        const sourceCodes = new Set(sourceOptions.map((item) => item.code));
        const signature = [
            sourceOptions.map((i) => `${i.code}:${i.label}`).join("|"),
            normalized.map((i) => `${i.code}:${i.label}`).join("|"),
        ].join("//");
        const sourcePreferred = sourceCodes.has(sourceLangSel.value) ? sourceLangSel.value : (h.source_lang || "en");
        const targetPreferred = supportedCodes.has(targetLangSel.value) ? targetLangSel.value : (h.target_lang || "zh");

        const rebuildSelect = (select, items, codes, preferred, fallback) => {
            if (signature !== languageOptionSignature) {
                select.textContent = "";
                items.forEach((item) => {
                    select.appendChild(new Option(item.label, item.code));
                });
            }
            select.value = codes.has(preferred) ? preferred : fallback;
            if (!codes.has(select.value)) select.value = items[0].code;
        };

        rebuildSelect(sourceLangSel, sourceOptions, sourceCodes, sourcePreferred, "en");
        rebuildSelect(targetLangSel, normalized, supportedCodes, targetPreferred, "zh");
        languageOptionSignature = signature;
        updateSwapAvailability();
    }

    /** 源语言是「自动检测」时不能交换语言方向。 */
    function updateSwapAvailability() {
        if (!swapLangBtn) return;
        const auto = sourceLangSel.value === "auto";
        swapLangBtn.disabled = auto || isRecording || isStarting;
        swapLangBtn.title = auto ? "源语言为自动检测时无法交换" : "交换语言";
    }

    function latencyModeLabel(mode) {
        return {
            low: "低延迟",
            balanced: "均衡",
            stable: "稳妥",
        }[mode] || "均衡";
    }

    function mtWarningLabel(warning, truncated = false) {
        if (truncated || warning === "mt_output_truncated") return "译文已截断";
        if (warning === "mt_timeout") return "翻译超时";
        if (warning === "client_disconnected") return "连接中断";
        if (warning === "mt_output_cleaned") return "译文已清理";
        return "";
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
            if (typeof prefs.persistHistory === "boolean") {
                persistHistoryChk.checked = prefs.persistHistory;
            }
            if (autoUnloadChk && typeof prefs.autoUnload === "boolean") {
                autoUnloadChk.checked = prefs.autoUnload;
            }
            if (prefs.audioSource && optionExists(audioSourceSel, prefs.audioSource)) {
                audioSourceSel.value = prefs.audioSource;
            }
            if (prefs.latencyMode && optionExists(latencyModeSel, prefs.latencyMode)) {
                latencyModeSel.value = prefs.latencyMode;
            }
            if (typeof prefs.streamingPreview === "boolean") {
                streamingPreviewChk.checked = prefs.streamingPreview;
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
            persistHistory: persistHistoryChk.checked,
            autoUnload: autoUnloadChk ? autoUnloadChk.checked : true,
            audioSource: audioSourceSel.value,
            latencyMode: latencyModeSel.value,
            streamingPreview: streamingPreviewChk.checked,
            subtitleMode: subtitleModeSel.value,
            glossary: glossary,
        }));
    }

    function historyEnabled() {
        return persistHistoryChk ? persistHistoryChk.checked : true;
    }

    function autoUnloadEnabled() {
        return autoUnloadChk ? autoUnloadChk.checked : true;
    }

    function hasLoadedModels() {
        return !!(lastHealth?.asr_loaded || lastHealth?.mt_loaded);
    }

    function modelActionText() {
        if (isDownloading) return "加载/下载中...";
        if (backendLoading) return "加载中...";
        if (backendReady) return "模型已就绪";
        return "加载/下载模型";
    }

    function modelActionTitle() {
        if (backendReady) return "模型已经加载完成；如需释放内存，请使用释放内存按钮。";
        return "加载已缓存模型；如果本机缺少模型，会按配置下载。";
    }

    function clearAutoUnloadTimer() {
        if (autoUnloadTimer) clearTimeout(autoUnloadTimer);
        autoUnloadTimer = 0;
    }

    function autoUnloadBlocked() {
        return !autoUnloadEnabled() || isRecording || isStarting || isDownloading ||
            isUnloadingModels || backendLoading || !hasLoadedModels();
    }

    function markUserActivity() {
        lastUserActivityAt = Date.now();
        scheduleAutoUnload();
    }

    function scheduleAutoUnload() {
        clearAutoUnloadTimer();
        if (autoUnloadBlocked()) return;
        const remaining = AUTO_UNLOAD_IDLE_MS - (Date.now() - lastUserActivityAt);
        if (remaining <= 0) {
            autoUnloadModels();
            return;
        }
        autoUnloadTimer = setTimeout(autoUnloadModels, remaining);
    }

    async function autoUnloadModels() {
        clearAutoUnloadTimer();
        if (autoUnloadBlocked()) return;
        await unloadModels({ automatic: true });
    }

    function renderHistoryInfo() {
        if (!historyInfo) return;
        const suffix = historyEnabled() ? "" : " · 未保存";
        const serverHistory = lastHealth?.transcript_history;
        const sizeSuffix = serverHistory?.enabled && serverHistory.total_bytes
            ? ` · ${fmtBytes(serverHistory.total_bytes)}`
            : "";
        const finalizedCount = [...segments.values()].filter((entry) => !entry.isPreview).length;
        historyInfo.textContent = `记录 ${finalizedCount}/${MAX_SEGMENTS}${suffix}${sizeSuffix}`;
        if (!historyEnabled()) {
            historyInfo.title = "保存记录已关闭：当前浏览器缓存和本轮后端落盘都会停止";
        } else {
            historyInfo.title = serverHistory?.enabled
                ? `本机保留最近 ${MAX_SEGMENTS} 段，刷新后可恢复；后端自动落盘：${serverHistory.history_dir || ""}`
                : `本机保留最近 ${MAX_SEGMENTS} 段，刷新后可恢复`;
        }
    }

    function currentOriginalSegments() {
        return [...segments.entries()]
            .filter(([, entry]) => !entry.isPreview)
            .map(([id, entry]) => ({
                segment_id: id,
                text: String(entry.originalText || "").trim(),
                source_lang: entry.srcLang,
                target_lang: entry.tgtLang,
            }))
            .filter((item) => item.text);
    }

    function serializeEntry(id, entry) {
        return {
            segment_id: id,
            ts: entry.ts?.toISOString?.() || new Date().toISOString(),
            source_lang: entry.srcLang,
            target_lang: entry.tgtLang,
            original_text: entry.originalText || "",
            translated_text: entry.translatedText || "",
            audio_ms: entry.audioMs,
            asr_ms: entry.asrMs,
            mt_ms: entry.mtMs,
            total_ms: entry.totalMs,
            rtf: entry.rtf,
            warning: entry.warning || "",
            truncated: !!entry.truncated,
            state: entry.translatedRow.classList.contains("error") ? "error" :
                (entry.translatedRow.classList.contains("streaming") ? "streaming" : "final"),
        };
    }

    function saveHistory() {
        clearTimeout(historySaveTimer);
        historySaveTimer = 0;
        if (!historyEnabled()) {
            localStorage.removeItem(HISTORY_KEY);
            renderHistoryInfo();
            return;
        }

        const items = [...segments.entries()]
            .filter(([, entry]) => !entry.isPreview)
            .slice(-MAX_SEGMENTS)
            .map(([id, entry]) => serializeEntry(id, entry));
        try {
            localStorage.setItem(HISTORY_KEY, JSON.stringify({
                saved_at: new Date().toISOString(),
                max_segments: MAX_SEGMENTS,
                items,
            }));
        } catch {
            const compactItems = items.slice(-Math.floor(MAX_SEGMENTS / 2));
            try {
                localStorage.setItem(HISTORY_KEY, JSON.stringify({
                    saved_at: new Date().toISOString(),
                    max_segments: compactItems.length,
                    items: compactItems,
                }));
                flashHint("记录过多，已仅保留最近一部分");
            } catch {
                localStorage.removeItem(HISTORY_KEY);
                flashHint("本机记录保存失败，已清理缓存");
            }
        }
        renderHistoryInfo();
    }

    function scheduleHistorySave() {
        if (isRestoringHistory) return;
        clearTimeout(historySaveTimer);
        historySaveTimer = setTimeout(saveHistory, HISTORY_SAVE_DEBOUNCE_MS);
    }

    function removeSegmentCard(id) {
        const entry = segments.get(id);
        if (entry?.card?.parentNode) entry.card.remove();
        segments.delete(id);
    }

    function trimSegments() {
        while (segments.size > MAX_SEGMENTS) {
            const oldest = segments.keys().next().value;
            if (oldest === undefined) break;
            removeSegmentCard(oldest);
        }
        if (segments.size === 0 && emptyState && !emptyState.parentNode) {
            transcript.appendChild(emptyState);
        }
        renderHistoryInfo();
    }

    function restoreHistory() {
        if (!historyEnabled()) {
            localStorage.removeItem(HISTORY_KEY);
            renderHistoryInfo();
            return;
        }
        let items = [];
        try {
            const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || "null");
            items = Array.isArray(saved?.items) ? saved.items.slice(-MAX_SEGMENTS) : [];
        } catch {
            localStorage.removeItem(HISTORY_KEY);
            renderHistoryInfo();
            return;
        }
        if (!items.length) {
            renderHistoryInfo();
            return;
        }

        isRestoringHistory = true;
        try {
            for (const item of items) {
                const id = Number(item.segment_id);
                if (!Number.isFinite(id)) continue;
                const entry = ensureCard(id, item.source_lang, item.target_lang);
                const ts = new Date(item.ts || Date.now());
                entry.ts = Number.isNaN(ts.getTime()) ? new Date() : ts;
                entry.tsEl.textContent = fmtTime(entry.ts);
                entry.tsEl.dateTime = entry.ts.toISOString();
                entry.originalText = item.original_text || "";
                entry.translatedText = item.translated_text || "";
                entry.original.textContent = entry.originalText;
                entry.translated.textContent = entry.translatedText || "";
                entry.audioMs = item.audio_ms ?? null;
                entry.asrMs = item.asr_ms ?? null;
                entry.mtMs = item.mt_ms ?? null;
                entry.totalMs = item.total_ms ?? null;
                entry.rtf = item.rtf ?? null;
                entry.warning = item.warning || "";
                entry.truncated = !!item.truncated;
                entry.translatedRow.classList.remove("pending", "streaming", "error");
                if (item.state === "error") entry.translatedRow.classList.add("error");
                if (!entry.translatedText) entry.translatedRow.classList.add("pending");
                setMetrics(entry);
                lastSegId = Math.max(lastSegId, id);
            }
        } finally {
            isRestoringHistory = false;
        }
        trimSegments();
        updateControls();
        scrollToLatest();
    }

    function historyRecordDate(record) {
        const raw = record.updated_at_iso || record.created_at_iso ||
            (record.updated_at ? record.updated_at * 1000 : null) ||
            (record.created_at ? record.created_at * 1000 : null);
        const d = raw ? new Date(raw) : new Date();
        return Number.isNaN(d.getTime()) ? new Date() : d;
    }

    function fmtHistoryDate(record) {
        const d = historyRecordDate(record);
        const p = (n) => String(n).padStart(2, "0");
        return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
    }

    function normalizeSavedRecord(record) {
        if (!record || typeof record !== "object") return null;
        const original = String(record.original_text || "");
        const translated = String(record.translated_text || "");
        if (!original && !translated) return null;
        return {
            ...record,
            original_text: original,
            translated_text: translated,
            source_lang: record.source_lang || "",
            target_lang: record.target_lang || "",
            state: record.state || "final",
        };
    }

    function filteredSavedRecords() {
        const query = (historySearch?.value || "").trim().toLocaleLowerCase();
        const state = historyStateFilter?.value || "all";
        return savedTranscriptRecords.filter((record) => {
            if (state !== "all" && record.state !== state) return false;
            if (!query) return true;
            const haystack = [
                record.original_text,
                record.translated_text,
                langLabel(record.source_lang),
                langLabel(record.target_lang),
                record.source_lang,
                record.target_lang,
            ].join(" ").toLocaleLowerCase();
            return haystack.includes(query);
        });
    }

    function renderSavedRecords() {
        if (!historyList || !historyStatus) return;
        const count = savedTranscriptRecords.length;
        visibleSavedTranscriptRecords = filteredSavedRecords();
        const visibleCount = visibleSavedTranscriptRecords.length;
        if (isHistoryLoading) {
            historyStatus.textContent = "读取中...";
        } else if (isHistoryClearing) {
            historyStatus.textContent = "清理中...";
        } else if (count && visibleCount !== count) {
            historyStatus.textContent = `显示 ${visibleCount} / ${count} 条`;
        } else {
            historyStatus.textContent = count ? `最近 ${count} 条` : "暂无记录";
        }
        if (!count) {
            historyList.innerHTML = '<div class="history-empty">暂无历史记录</div>';
            updateControls();
            return;
        }
        if (!visibleCount) {
            historyList.innerHTML = '<div class="history-empty"><strong>没有匹配记录</strong><br>换个关键词或筛选条件试试</div>';
            updateControls();
            return;
        }

        historyList.innerHTML = visibleSavedTranscriptRecords.map((record, index) => {
            const stateLabel = record.state === "error" ? "错误" : "完成";
            const lang = `${langLabel(record.source_lang)} → ${langLabel(record.target_lang)}`;
            return `
                <div class="history-item" data-history-index="${index}">
                    <div class="history-meta">
                        <strong>${escHtml(fmtHistoryDate(record))}</strong>
                        <span>${escHtml(lang)}</span>
                        <span>${escHtml(stateLabel)}</span>
                    </div>
                    <div class="history-copy">
                        <p class="history-original">${escHtml(record.original_text)}</p>
                        <p>${escHtml(record.translated_text)}</p>
                        <div class="history-item-actions">
                            <button type="button" class="btn-secondary history-inline-btn" data-history-action="load">载入</button>
                            <button type="button" class="btn-secondary history-inline-btn" data-history-action="copy">复制</button>
                        </div>
                    </div>
                </div>
            `;
        }).join("");
        updateControls();
    }

    async function fetchRecentTranscripts(showFlash = false) {
        if (isHistoryLoading) return;
        isHistoryLoading = true;
        renderSavedRecords();
        updateControls();
        let failed = false;
        try {
            const response = await apiFetch("/api/transcripts/recent?limit=80", { cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            savedTranscriptRecords = (Array.isArray(data.items) ? data.items : [])
                .map(normalizeSavedRecord)
                .filter(Boolean);
            renderSavedRecords();
            if (showFlash) flashHint(savedTranscriptRecords.length ? "历史记录已刷新" : "暂无历史记录");
        } catch (err) {
            failed = true;
            savedTranscriptRecords = [];
            visibleSavedTranscriptRecords = [];
            if (historyStatus) historyStatus.textContent = "读取失败";
            if (historyList) {
                historyList.innerHTML = `<div class="history-empty">读取失败：${escHtml(err.message || err)}</div>`;
            }
            flashHint("历史记录读取失败");
        } finally {
            isHistoryLoading = false;
            if (!failed) renderSavedRecords();
            updateControls();
        }
    }

    async function clearSavedTranscripts() {
        if (isRecording || isStarting) {
            flashHint("同传运行中，停止后再清空保存历史");
            return;
        }
        if (isHistoryClearing) return;
        const message = "确定清空已保存的历史记录吗？这不会影响当前屏幕上的同传内容。";
        if (!window.confirm(message)) return;

        isHistoryClearing = true;
        renderSavedRecords();
        updateControls();
        try {
            const response = await apiFetch("/api/transcripts/clear", { method: "POST" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            savedTranscriptRecords = [];
            visibleSavedTranscriptRecords = [];
            renderSavedRecords();
            await pollHealth();
            flashHint("已清空保存历史");
        } catch (err) {
            flashHint("清空保存历史失败：" + (err.message || err));
        } finally {
            isHistoryClearing = false;
            renderSavedRecords();
            updateControls();
        }
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

    function syncIdleHint(mtCpuOnly = false) {
        if (isRecording || isStarting) return;
        if (backendLoading) {
            setHint("模型加载中，请稍候...");
        } else if (!backendReady) {
            setHint("模型未就绪，点击「加载/下载模型」获取或检查后端日志");
        } else if (mtCpuOnly) {
            setHint("MT 正在 CPU 模式运行，可用但延迟较高");
        } else {
            setHint(idleHintText());
        }
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
        updateSwapAvailability();
        clearBtn.disabled = !hasSegments;
        copyBtn.disabled = !hasSegments;
        exportTxtBtn.disabled = !hasSegments;
        exportSrtBtn.disabled = !hasSegments;
        if (fullTextGenerateBtn) {
            // 本地 MT 是单实例串行的：同传运行时再发全文翻译只会排队阻塞，
            // 体验上像卡死，所以直接禁用并说明原因。
            const busyWithLiveMt = isRecording || isStarting;
            fullTextGenerateBtn.disabled = isFullTextTranslating || busyWithLiveMt ||
                currentOriginalSegments().length === 0;
            fullTextGenerateBtn.textContent = isFullTextTranslating ? "生成中..." : "生成全文翻译";
            fullTextGenerateBtn.title = busyWithLiveMt
                ? "同传运行中不能生成全文翻译，请先停止同传"
                : "基于本次全部原文，用全文上下文重新翻译";
        }
        const hasFullTextResult = !!(
            fullTextResult?.full_original_text && fullTextResult?.full_translated_text
        );
        if (fullTextCopyBtn) fullTextCopyBtn.disabled = isFullTextTranslating || !hasFullTextResult;
        if (fullTextExportBtn) fullTextExportBtn.disabled = isFullTextTranslating || !hasFullTextResult;
        if (historyRefreshBtn) historyRefreshBtn.disabled = isHistoryLoading;
        if (historyCopyFilteredBtn) {
            historyCopyFilteredBtn.disabled = isHistoryLoading || isHistoryClearing ||
                visibleSavedTranscriptRecords.length === 0;
        }
        if (historyExportFilteredBtn) {
            historyExportFilteredBtn.disabled = isHistoryLoading || isHistoryClearing ||
                visibleSavedTranscriptRecords.length === 0;
        }
        if (historyLoadBtn) {
            historyLoadBtn.disabled = isRecording || isStarting || isHistoryLoading || isHistoryClearing ||
                visibleSavedTranscriptRecords.length === 0;
        }
        if (historyClearSavedBtn) {
            historyClearSavedBtn.disabled = isRecording || isStarting || isHistoryLoading || isHistoryClearing ||
                savedTranscriptRecords.length === 0;
        }
        modelDownloadBtn.disabled = isDownloading || isUnloadingModels || isRecording || isStarting || backendLoading || backendReady;
        if (modelUnloadBtn) {
            modelUnloadBtn.disabled = isDownloading || isUnloadingModels || isRecording || isStarting ||
                backendLoading || !hasLoadedModels();
            modelUnloadBtn.textContent = isUnloadingModels ? "释放中..." : "释放内存";
        }
        if (autoUnloadChk) {
            autoUnloadChk.disabled = isRecording || isStarting || isDownloading || isUnloadingModels;
        }
        audioSourceSel.disabled = isRecording || isStarting;
        latencyModeSel.disabled = isStarting;
        streamingPreviewChk.disabled = isStarting;
        const downloadText = modelActionText();
        modelDownloadBtn.textContent = downloadText;
        modelDownloadBtn.title = modelActionTitle();
        if (modelPromptDownload) {
            modelPromptDownload.disabled = modelDownloadBtn.disabled;
            modelPromptDownload.textContent = downloadText;
            modelPromptDownload.title = modelActionTitle();
        }
        renderHistoryInfo();
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
                : (shortModelName(health?.mt_model_id) || "Hy-MT2-1.8B-Q4_K_M.gguf");
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

    function clampSubtitleNumber(value, fallback, min, max) {
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return fallback;
        return Math.min(max, Math.max(min, Math.round(parsed)));
    }

    function readSubtitleJson(key) {
        try {
            const parsed = JSON.parse(localStorage.getItem(key) || "null");
            return parsed && typeof parsed === "object" ? parsed : null;
        } catch {
            return null;
        }
    }

    function loadSubtitleWindowState() {
        const size = readSubtitleJson(SUBTITLE_FRAME_SIZE_KEY) || {};
        const position = readSubtitleJson(SUBTITLE_WINDOW_POS_KEY) || {};
        const state = {
            width: clampSubtitleNumber(size.width, 960, 420, 1800),
            height: clampSubtitleNumber(size.height, 260, 140, 900),
        };
        const x = Number(position.x);
        const y = Number(position.y);
        if (Number.isFinite(x) && Number.isFinite(y)) {
            state.x = clampSubtitleNumber(x, 0, -5000, 10000);
            state.y = clampSubtitleNumber(y, 0, -5000, 10000);
        }
        return state;
    }

    function subtitlePopupFeatures(state) {
        const parts = [
            `width=${state.width}`,
            `height=${state.height}`,
            "resizable=yes",
            "menubar=no",
            "toolbar=no",
            "location=no",
            "status=no",
        ];
        if (Number.isFinite(Number(state.x)) && Number.isFinite(Number(state.y))) {
            parts.push(`left=${state.x}`, `top=${state.y}`);
        }
        return parts.join(",");
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
        const overlayUrl = `/overlay.html?mode=${encodeURIComponent(mode)}#token=${encodeURIComponent(apiToken)}`;
        const windowState = loadSubtitleWindowState();
        let opened = false;

        if (window.pywebview?.api?.open_subtitle_window) {
            try {
                const result = await desktopCall("open_subtitle_window", mode, windowState);
                opened = !!result?.ok;
            } catch {}
        }

        if (!opened) {
            const popup = window.open(
                overlayUrl,
                "translive_subtitles",
                subtitlePopupFeatures(windowState),
            );
            opened = !!popup;
        }

        if (opened) syncClickThroughVisibility();
        flashHint(opened ? "字幕窗已打开" : "字幕窗被系统拦截，请允许弹窗");
    }

    /** 点击穿透只有桌面版（原生 NSWindow）才有，浏览器里没有对应能力。 */
    function syncClickThroughVisibility() {
        if (!clickThroughWrap) return;
        const supported = !!window.pywebview?.api?.set_subtitle_click_through;
        clickThroughWrap.hidden = !supported;
    }

    async function applyClickThrough() {
        if (!clickThroughToggle || !window.pywebview?.api?.set_subtitle_click_through) return;
        const enabled = clickThroughToggle.checked;
        try {
            const result = await desktopCall("set_subtitle_click_through", enabled);
            if (!result?.ok) {
                clickThroughToggle.checked = false;
                flashHint(result?.error === "subtitle_window_not_open"
                    ? "请先打开字幕窗"
                    : "点击穿透设置失败");
                return;
            }
            flashHint(enabled
                ? "字幕正文已可点击穿透，工具栏和缩放手柄不受影响"
                : "字幕窗恢复完全可点击");
        } catch (err) {
            clickThroughToggle.checked = false;
            flashHint("点击穿透设置失败：" + (err.message || err));
        }
    }

    async function clearSubtitleWindow(options = {}) {
        const showFlash = !!options.showFlash;
        postSubtitleControl({ type: "clear" });
        try {
            const response = await apiFetch("/api/subtitles/clear", { method: "POST" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            if (showFlash) flashHint("当前字幕已清空");
            return true;
        } catch (err) {
            if (showFlash) flashHint("字幕窗同步清空失败：" + (err.message || err));
            return false;
        }
    }

    function scheduleSubtitleMirrorReconnect() {
        if (captureMode !== "system" || !isRecording || subtitleMirrorReconnectTimer) return;
        if (subtitleMirrorReconnectAttempts >= MAX_SUBTITLE_MIRROR_RECONNECT_ATTEMPTS) {
            flashHint("字幕同步连接恢复失败，系统音频仍在运行");
            return;
        }

        const delay = Math.min(8000, 600 * Math.pow(2, subtitleMirrorReconnectAttempts));
        subtitleMirrorReconnectAttempts += 1;
        subtitleMirrorReconnectTimer = setTimeout(async () => {
            subtitleMirrorReconnectTimer = 0;
            try {
                await connectSubtitleMirror();
                subtitleMirrorReconnectAttempts = 0;
                flashHint("字幕同步已恢复");
            } catch {
                scheduleSubtitleMirrorReconnect();
            }
        }, delay);
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
                    warning: segment.warning,
                    truncated: segment.truncated,
                });
            }
        });
    }

    function applySubtitleMirrorMessage(payload) {
        if (!payload || typeof payload !== "object") return;
        if (payload.type === "snapshot") applySubtitleSnapshot(payload);
        else if (payload.type === "clear") {
            segments.clear();
            clearFullTextResult();
            localStorage.removeItem(HISTORY_KEY);
            transcript.innerHTML = "";
            transcript.appendChild(emptyState);
            userScrolledAwayFromLatest = false;
            jumpPill.hidden = true;
            updateControls();
        }
        else if (payload.type === "original_preview") applyOriginalPreview(payload);
        else if (payload.type === "original_preview_clear") applyOriginalPreviewClear(payload);
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
            subtitleMirrorWs = new WebSocket(mirrorUrl, wsProtocols);

            subtitleMirrorWs.onopen = () => {
                settled = true;
                subtitleMirrorReconnectAttempts = 0;
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
                    return;
                }
                if (captureMode === "system" && isRecording) {
                    scheduleSubtitleMirrorReconnect();
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
        clearSubtitleMirrorReconnect();
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
            warning: "",
            truncated: false,
            isPreview: false,
            previewMs: null,
        };

        entry.tsEl.textContent = fmtTime(now);
        entry.tsEl.dateTime = now.toISOString();
        setLangPair(entry, entry.srcLang, entry.tgtLang);

        segments.set(id, entry);
        transcript.prepend(node);
        trimSegments();
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
        if (entry.isPreview && entry.previewMs != null) parts.push(`预览 ${entry.previewMs}ms`);
        const warning = mtWarningLabel(entry.warning, entry.truncated);
        if (warning) parts.push(warning);
        entry.metrics.textContent = parts.join(" · ");
    }

    function applyOriginal(payload) {
        const entry = ensureCard(
            payload.segment_id,
            payload.source_lang || sourceLangSel.value,
            payload.target_lang || targetLangSel.value,
        );
        entry.isPreview = false;
        entry.previewMs = null;
        entry.card.classList.remove("preview");
        entry.originalText = payload.text || "";
        entry.original.textContent = entry.originalText;
        if (payload.audio_duration_ms != null) entry.audioMs = payload.audio_duration_ms;
        if (payload.asr_ms != null) entry.asrMs = payload.asr_ms;
        setMetrics(entry);
        scrollToLatest();
        scheduleHistorySave();
        if (fullTextResult && !isFullTextTranslating) clearFullTextResult();
    }

    function applyOriginalPreview(payload) {
        const existing = segments.get(payload.segment_id);
        if (existing && !existing.isPreview && existing.originalText) return;
        const entry = ensureCard(
            payload.segment_id,
            payload.source_lang || sourceLangSel.value,
            payload.target_lang || targetLangSel.value,
        );
        entry.isPreview = true;
        entry.previewMs = payload.preview_ms ?? null;
        entry.card.classList.add("preview");
        entry.originalText = payload.text || "";
        entry.original.textContent = entry.originalText;
        setMetrics(entry);
        scrollToLatest();
    }

    function applyOriginalPreviewClear(payload) {
        const id = Number(payload.segment_id);
        const entry = segments.get(id);
        if (!entry?.isPreview) return;
        removeSegmentCard(id);
        trimSegments();
        updateControls();
    }

    function applyTranslatedPartial(payload) {
        const existing = segments.get(payload.segment_id);
        const entry = ensureCard(
            payload.segment_id,
            existing?.srcLang || sourceLangSel.value,
            payload.target_lang || existing?.tgtLang || targetLangSel.value,
        );
        entry.isPreview = false;
        entry.previewMs = null;
        entry.card.classList.remove("preview");
        entry.translatedText = payload.accumulated || payload.text || "";
        entry.translated.textContent = entry.translatedText;
        entry.translatedRow.classList.remove("pending", "error");
        entry.translatedRow.classList.add("streaming");
        scrollToLatest();
        scheduleHistorySave();
    }

    function applyTranslated(payload) {
        const existing = segments.get(payload.segment_id);
        const entry = ensureCard(
            payload.segment_id,
            payload.source_lang || existing?.srcLang || sourceLangSel.value,
            payload.target_lang || existing?.tgtLang || targetLangSel.value,
        );
        entry.isPreview = false;
        entry.previewMs = null;
        entry.card.classList.remove("preview");
        entry.translatedText = payload.text || "";
        entry.translated.textContent = entry.translatedText;
        entry.translatedRow.classList.remove("pending", "error", "streaming");
        if (payload.audio_duration_ms != null) entry.audioMs = payload.audio_duration_ms;
        if (payload.asr_ms != null) entry.asrMs = payload.asr_ms;
        if (payload.mt_ms != null) entry.mtMs = payload.mt_ms;
        if (payload.total_ms != null) entry.totalMs = payload.total_ms;
        if (payload.rtf != null) entry.rtf = payload.rtf;
        entry.warning = payload.warning || "";
        entry.truncated = !!payload.truncated;
        setMetrics(entry);
        scrollToLatest();
        scheduleHistorySave();
    }

    function applyError(payload) {
        const entry = ensureCard(payload.segment_id, sourceLangSel.value, targetLangSel.value);
        entry.isPreview = false;
        entry.previewMs = null;
        entry.card.classList.remove("preview");
        entry.translatedText = "[Error] " + (payload.text || "未知错误");
        entry.translated.textContent = entry.translatedText;
        entry.translatedRow.classList.remove("pending", "streaming");
        entry.translatedRow.classList.add("error");
        setMetrics(entry);
        scrollToLatest();
        scheduleHistorySave();
    }

    function nearLatest() {
        return transcript.scrollTop < 48;
    }

    function scrollToLatest() {
        if (!autoScrollChk.checked) return;
        if (userScrolledAwayFromLatest && !nearLatest()) return;
        transcript.scrollTop = 0;
    }

    transcript.addEventListener("scroll", () => {
        userScrolledAwayFromLatest = !nearLatest();
        jumpPill.hidden = !userScrolledAwayFromLatest;
    });

    jumpPill.addEventListener("click", () => {
        userScrolledAwayFromLatest = false;
        transcript.scrollTop = 0;
        jumpPill.hidden = true;
    });

    function sessionConfig() {
        return {
            source_lang: sourceLangSel.value,
            target_lang: targetLangSel.value,
            latency_mode: latencyModeSel.value,
            persist_history: historyEnabled(),
            streaming_preview: streamingPreviewChk.checked,
            glossary: glossaryObject(),
        };
    }

    function sendConfig() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const config = sessionConfig();
        ws.send(JSON.stringify({
            type: "config",
            source_lang: config.source_lang,
            target_lang: config.target_lang,
            latency_mode: config.latency_mode,
            persist_history: config.persist_history,
            streaming_preview: config.streaming_preview,
        }));
        sendGlossaryToServer();
    }

    async function sendSystemAudioConfig() {
        if (captureMode !== "system" || !isRecording) return;
        if (!window.pywebview?.api?.update_system_audio_config) return;
        try {
            const result = await desktopCall("update_system_audio_config", sessionConfig());
            if (!result?.ok) {
                flashHint("系统音频语言设置同步失败：" + (result?.error || "未知错误"));
            }
        } catch (err) {
            flashHint("系统音频语言设置同步失败：" + (err.message || err));
        }
    }

    function scheduleReconnect(options = {}) {
        const resumeMode = options.resumeMode || null;
        if (!backendReady || reconnectAttempts >= MAX_RECONNECT_ATTEMPTS || reconnectTimer) {
            if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
                setStatus("未连接");
                setHint("连接多次恢复失败，点击「开始同传」重试");
                flashHint("连接恢复失败，请手动重试");
            }
            return;
        }

        const delay = Math.min(8000, 600 * Math.pow(2, reconnectAttempts));
        reconnectAttempts += 1;
        setStatus("重连中", "connecting");
        setHint(resumeMode === "mic"
            ? `连接断开，${Math.ceil(delay / 1000)} 秒后自动恢复麦克风...`
            : `连接断开，${Math.ceil(delay / 1000)} 秒后重连...`);

        reconnectTimer = setTimeout(async () => {
            reconnectTimer = 0;
            try {
                await connect();
                reconnectAttempts = 0;
                if (resumeMode === "mic") {
                    const resumed = await startMicrophoneRecording({ fromReconnect: true, skipConnect: true });
                    if (!resumed) {
                        setStatus("已连接", "connected");
                        setHint("连接已恢复，但麦克风无法自动重新打开");
                        return;
                    }
                    flashHint("连接已恢复，已继续同传");
                } else {
                    setHint("连接已恢复，点击「开始同传」继续");
                }
            } catch {
                scheduleReconnect({ resumeMode });
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
            ws = new WebSocket(wsUrl, wsProtocols);
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
                if (pendingSessionStop) {
                    pendingSessionStop();
                    pendingSessionStop = null;
                }
                const wasRecording = isRecording;
                const previousMode = captureMode;
                ws = null;
                if (!opened && !settled) {
                    settled = true;
                    reject(new Error("WebSocket 连接失败"));
                }
                if (wasRecording) {
                    stopRecording({
                        closeSocket: false,
                        hintText: previousMode === "mic"
                            ? "连接已断开，正在尝试自动恢复麦克风"
                            : "连接已断开，正在尝试重连",
                        keepReconnect: true,
                    });
                    scheduleReconnect({ resumeMode: previousMode === "mic" ? "mic" : null });
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
                if (data.type === "session_stopped") {
                    if (pendingSessionStop) {
                        pendingSessionStop();
                        pendingSessionStop = null;
                    }
                }
                else if (data.type === "original_preview") applyOriginalPreview(data);
                else if (data.type === "original_preview_clear") applyOriginalPreviewClear(data);
                else if (data.type === "original") applyOriginal(data);
                else if (data.type === "translated_partial") applyTranslatedPartial(data);
                else if (data.type === "translated") applyTranslated(data);
                else if (data.type === "error") applyError(data);
            };
        }).finally(() => {
            pendingConnect = null;
        });

        return pendingConnect;
    }

    async function startRecording(options = {}) {
        if (isRecording) return true;
        if (isStarting) return false;
        clearAutoUnloadTimer();
        if (shouldUseSystemAudio()) {
            return startSystemAudioRecording(options);
        }
        return startMicrophoneRecording(options);
    }

    async function startMicrophoneRecording(options = {}) {
        const fromReconnect = !!options.fromReconnect;
        const skipConnect = !!options.skipConnect;
        clearReconnect();
        isStarting = true;
        updateControls();
        setStatus("准备中", "connecting");
        setHint(fromReconnect ? "连接已恢复，正在重新打开麦克风..." : "正在准备麦克风...");
        let started = false;

        try {
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error("当前浏览器不支持麦克风采集");
            }

            if (!skipConnect || !ws || ws.readyState !== WebSocket.OPEN) await connect();

            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

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
            started = true;
        } catch (err) {
            cleanupAudio();
            if (ws && ws.readyState !== WebSocket.CLOSED) {
                try { ws.close(); } catch {}
            }
            setStatus("未连接");
            setHint(idleHintText());
            if (fromReconnect) {
                flashHint("自动恢复麦克风失败：" + (err.message || err));
            } else {
                flashHint("启动失败：" + (err.message || err));
            }
        } finally {
            isStarting = false;
            updateControls();
        }
        return started;
    }

    function shouldUseSystemAudio() {
        return audioSourceSel.value === "system" &&
            !!window.pywebview?.api?.start_system_audio_capture &&
            nativeAudioAvailable;
    }

    async function startSystemAudioRecording() {
        clearReconnect();
        clearSubtitleMirrorReconnect();
        isStarting = true;
        updateControls();
        setStatus("准备中", "connecting");
        setHint("正在准备系统音频采集...");
        setAudioSourceStatus("启动中", "warn", "正在启动系统音频采集");
        let started = false;

        try {
            await connectSubtitleMirror();
            await apiFetch("/api/subtitles/clear", { method: "POST" }).catch(() => {});
            const result = await desktopCall("start_system_audio_capture", sessionConfig());
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
            setAudioSourceStatus("采集中", "ok", "系统音频正在采集");
            started = true;
        } catch (err) {
            closeSubtitleMirror();
            try { await desktopCall("stop_system_audio_capture"); } catch {}
            setStatus("未连接");
            setHint(idleHintText());
            setAudioSourceStatus("启动失败", "error", err.message || String(err));
            flashHint("系统音频启动失败：" + (err.message || err));
        } finally {
            isStarting = false;
            updateControls();
        }
        return started;
    }

    function requestGracefulStreamStop(socket) {
        if (!socket || socket.readyState !== WebSocket.OPEN) return Promise.resolve();
        return new Promise((resolve) => {
            let settled = false;
            const finish = () => {
                if (settled) return;
                settled = true;
                clearTimeout(timer);
                resolve();
            };
            const timer = setTimeout(finish, 22000);
            pendingSessionStop = finish;
            try {
                socket.send(JSON.stringify({ type: "stop" }));
            } catch {
                finish();
            }
        });
    }

    async function stopRecording(options = {}) {
        const {
            closeSocket = true,
            hintText = "已停止，点击「开始同传」继续",
            keepReconnect = false,
        } = options;
        if (!keepReconnect) clearReconnect();
        isRecording = false;
        if (captureMode === "system") {
            try { await desktopCall("stop_system_audio_capture"); } catch {}
            closeSubtitleMirror();
            setAudioSourceStatus(nativeAudioAvailable ? "可用" : "不可用", nativeAudioAvailable ? "ok" : "warn", audioSourceSel.title);
        } else {
            cleanupAudio();
            if (closeSocket) await requestGracefulStreamStop(ws);
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
        lastUserActivityAt = Date.now();
        scheduleAutoUnload();
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
        let lastPaint = 0;
        const tick = (timestamp) => {
            if (!analyser) return;
            // 电平条 30fps 已足够流畅，避免在 120Hz 屏幕上无意义地翻倍计算。
            if (timestamp - lastPaint >= 33) {
                lastPaint = timestamp;
                analyser.getByteTimeDomainData(buf);
                let sum = 0;
                for (let i = 0; i < buf.length; i++) {
                    const v = (buf[i] - 128) / 128;
                    sum += v * v;
                }
                const rms = Math.sqrt(sum / buf.length);
                const pct = Math.min(100, Math.pow(rms * 2.2, 0.72) * 100);
                meterBar.style.width = pct.toFixed(1) + "%";
            }
            meterRaf = requestAnimationFrame(tick);
        };
        meterRaf = requestAnimationFrame(tick);
    }

    function stopMeter() {
        if (meterRaf) cancelAnimationFrame(meterRaf);
        meterRaf = 0;
    }

    function clearHealthPollTimer() {
        if (healthPollTimer) clearTimeout(healthPollTimer);
        healthPollTimer = 0;
    }

    function nextHealthPollDelay(lastOk = true) {
        if (!lastOk || navigator.onLine === false) return HEALTH_POLL_ERROR_MS;
        if (isRecording || isStarting || backendLoading) return HEALTH_POLL_ACTIVE_MS;
        if (document.hidden) return HEALTH_POLL_HIDDEN_MS;
        return HEALTH_POLL_IDLE_MS;
    }

    function scheduleHealthPoll(delayMs) {
        clearHealthPollTimer();
        const delay = Math.max(1000, Number(delayMs) || nextHealthPollDelay(true));
        healthPollTimer = setTimeout(runScheduledHealthPoll, delay);
    }

    async function runScheduledHealthPoll() {
        clearHealthPollTimer();
        const ok = await pollHealth();
        scheduleHealthPoll(nextHealthPollDelay(ok));
    }

    function pollHealth() {
        if (pendingHealthPoll) return pendingHealthPoll;
        pendingHealthPoll = doPollHealth().finally(() => {
            pendingHealthPoll = null;
        });
        return pendingHealthPoll;
    }

    async function doPollHealth() {
        try {
            const response = await fetchWithTimeout("/api/health", { cache: "no-store" });
            if (!response.ok) return false;
            const h = await response.json();
            lastHealth = h;
            syncLanguageOptions(h);
            if (versionPill && h.app_version) {
                versionPill.textContent = h.app_version.startsWith("v") ? h.app_version : `v${h.app_version}`;
            }
            if (creditLine && h.app_credit) creditLine.textContent = h.app_credit;
            if (h.app_name && h.app_version) {
                document.title = `${h.app_name} ${h.app_version} · AI 同声传译`;
            }

            const asrValue = [
                h.asr_backend,
                h.asr_device,
                h.asr_compute_type,
                shortModelName(h.asr_effective_model_id || h.asr_model_id || h.asr_model_size),
            ].filter(Boolean).join(" · ");
            const asrAccelNote = h.asr_acceleration?.note || "";
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
            const asrText = asrLoading ? "加载中..." : (h.asr_loaded ? (asrValue || "已加载") : "未加载");
            const mtText = mtLoading ? "加载中..." : (h.mt_loaded ? (mtValue || "已加载") : "未加载");

            setBadge(
                asrBadge, "ASR", asrText, asrState,
                asrAccelNote ? `ASR: ${asrText}。${asrAccelNote}` : `ASR: ${asrText}`,
            );
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
            }
            syncIdleHint(mtCpuOnly);
            scheduleAutoUnload();

            const p = h.perf_stats || {};
            const runtimeText = runtimeSummary(h.runtime);
            const subtitleDropped = Number(h.subtitle_hub?.dropped_messages || 0);
            const subtitleText = subtitleDropped > 0 ? `字幕丢弃 ${subtitleDropped}` : "";
            const asrSuppressed = Number(h.asr_filter?.suppressed_segments || 0);
            const asrFilterText = asrSuppressed > 0 ? `已滤噪声 ${asrSuppressed}` : "";
            const historyFailures = Number(h.transcript_history?.write_failures || 0);
            const historyFailureText = historyFailures > 0 ? `记录写入失败 ${historyFailures}` : "";
            const extraStats = [runtimeText, subtitleText, asrFilterText, historyFailureText].filter(Boolean).join(" · ");
            if (p.segments) {
                statsEl.textContent =
                    `段数 ${p.segments} · ASR ${Math.round(p.avg_asr_ms || 0)}ms` +
                    ` · MT ${Math.round(p.avg_mt_ms || 0)}ms · RTF ${(p.avg_rtf || 0).toFixed(2)}` +
                    (extraStats ? ` · ${extraStats}` : "");
            } else if (newLoading) {
                statsEl.textContent = extraStats ? `模型加载中... · ${extraStats}` : "模型加载中...";
            } else if (mtCpuOnly) {
                statsEl.textContent = extraStats ? `MT CPU 模式 · ${extraStats}` : "MT CPU 模式：可用，但延迟可能较高";
            } else {
                statsEl.textContent = extraStats || (h.mt_unavailable_reason ? "翻译模型未就绪" : "尚无统计数据");
            }
            await refreshNativeAudioStatus();
            return true;
        } catch {
            setBadge(asrBadge, "ASR", "无法读取", "warn", "无法读取后端状态");
            setBadge(mtBadge, "MT", "无法读取", "warn", "无法读取后端状态");
            statsEl.textContent = "后端状态不可用";
            if (backendReady || backendLoading) {
                backendReady = false;
                backendLoading = false;
                updateControls();
            }
            return false;
        }
    }

    async function downloadModels() {
        if (isDownloading) return;
        isDownloading = true;
        hideModelPrompt(false);
        updateControls();
        setHint("正在加载或下载模型...");
        let flashMessage = "";
        try {
            const needAsr = lastHealth ? !lastHealth.asr_loaded : true;
            const needMt = lastHealth ? !lastHealth.mt_loaded : true;
            const forceDownload = forceNextModelDownload;
            const response = await apiFetch("/api/models/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    asr: needAsr,
                    mt: needMt,
                    force: forceDownload,
                }),
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (data.asr_error || data.mt_error) {
                throw new Error(data.asr_error || data.mt_error);
            }
            await pollHealth();
            flashMessage = "模型已就绪";
            modelPromptDismissed = true;
            forceNextModelDownload = false;
        } catch (err) {
            forceNextModelDownload = true;
            flashMessage = "模型下载失败：" + (err.message || err) + "；再次点击会尝试重新下载修复。";
            modelPromptDismissed = false;
        } finally {
            isDownloading = false;
            setHint(isRecording ? (captureMode === "system" ? "正在监听系统音频..." : "正在监听麦克风...") : idleHintText());
            updateControls();
            if (flashMessage) flashHint(flashMessage);
        }
    }

    async function unloadModels(options = {}) {
        const automatic = !!options.automatic;
        if (isUnloadingModels || isRecording || isStarting) return;
        isUnloadingModels = true;
        clearAutoUnloadTimer();
        updateControls();
        setHint(automatic ? "空闲中，正在自动释放模型内存..." : "正在释放模型内存...");
        try {
            const response = await apiFetch("/api/models/unload", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ asr: true, mt: true }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok || data.ok === false) {
                throw new Error(data.message || data.error || `HTTP ${response.status}`);
            }
            backendReady = false;
            await pollHealth();
            flashHint(automatic ? "空闲已自动释放模型内存" : "模型内存已释放，需要时可重新加载模型");
        } catch (err) {
            flashHint((automatic ? "自动释放模型内存失败：" : "释放模型内存失败：") + (err.message || err));
        } finally {
            isUnloadingModels = false;
            setHint(backendReady ? idleHintText() :
                (automatic ? "模型已自动释放，需要时点击「加载/下载模型」重新加载" : "模型已释放，需要时点击「加载/下载模型」重新加载"));
            updateControls();
            scheduleAutoUnload();
        }
    }

    function buildPlain() {
        const lines = [];
        const ids = [...segments.entries()]
            .filter(([, entry]) => !entry.isPreview)
            .map(([id]) => id);
        for (const id of ids) {
            const e = segments.get(id);
            if (e.originalText) lines.push(`[${fmtTime(e.ts)}] (${e.srcLang}) ${e.originalText}`);
            if (e.translatedText) lines.push(`[${fmtTime(e.ts)}] (${e.tgtLang}) ${e.translatedText}`);
            lines.push("");
        }
        return lines.join("\n").trim();
    }

    function buildSrt() {
        const ids = [...segments.entries()]
            .filter(([, entry]) => !entry.isPreview)
            .map(([id]) => id);
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

    function fullTextWarningText(data) {
        const parts = [];
        if (data?.truncated) {
            const limits = data.limits || {};
            parts.push(`全文过长，已按最多 ${limits.max_chars || "规定字数"} 字截断`);
        }
        if (data?.warning === "empty_output") parts.push("模型未返回译文");
        return parts.join(" · ");
    }

    function renderFullTextResult(data = fullTextResult) {
        if (!fullTextBody || !fullTextStatus) return;
        if (isFullTextTranslating) {
            fullTextStatus.textContent = "全文翻译中...";
            fullTextBody.innerHTML = '<div class="fulltext-empty">正在合并本次全部原文并整体翻译，请稍候。</div>';
            return;
        }
        const originalText = String(data?.full_original_text || "").trim();
        const translatedText = String(data?.full_translated_text || "").trim();
        if (!originalText) {
            fullTextStatus.textContent = "将本次全部原文合并后整体翻译";
            fullTextBody.innerHTML = '<div class="fulltext-empty">当前没有全文翻译。点击「生成全文翻译」后，会把本次所有原文合并成一篇内容再统一翻译。</div>';
            return;
        }

        const warning = fullTextWarningText(data);
        const lang = `${langLabel(data.source_lang)} → ${langLabel(data.target_lang)}`;
        fullTextStatus.textContent = `已整体翻译 · ${lang}${warning ? " · " + warning : ""}`;
        fullTextBody.innerHTML = `
            <div class="fulltext-document">
                <section class="fulltext-document-block original">
                    <h3>完整原文</h3>
                    <p>${escHtml(originalText)}</p>
                </section>
                <section class="fulltext-document-block translated">
                    <h3>完整译文</h3>
                    <p>${escHtml(translatedText || "未返回译文")}</p>
                </section>
            </div>
        `;
    }

    function buildFullTextComparison() {
        const data = fullTextResult;
        const originalText = String(data?.full_original_text || "").trim();
        const translatedText = String(data?.full_translated_text || "").trim();
        if (!originalText || !translatedText) return "";
        const lines = [];
        lines.push(`全文翻译 ${langLabel(data.source_lang)} → ${langLabel(data.target_lang)}`);
        const warning = fullTextWarningText(data);
        if (warning) lines.push(`提示：${warning}`);
        lines.push("", "完整原文：", originalText, "", "完整译文：", translatedText);
        return lines.join("\n").trim();
    }

    async function generateFullTextTranslation() {
        if (isFullTextTranslating) return;
        const originals = currentOriginalSegments();
        if (!originals.length) {
            flashHint("当前没有可生成全文翻译的原文");
            return;
        }
        const fullOriginalText = originals.map((item) => item.text).join("\n").trim();
        if (fullTextPanel) {
            fullTextPanel.hidden = false;
            fullTextToggle?.classList.add("active");
        }
        isFullTextTranslating = true;
        renderFullTextResult();
        updateControls();
        setHint("正在合并全部原文并整体翻译...");
        let failed = false;
        try {
            const timeoutMs = Math.min(
                300000,
                Math.max(120000, 90000 + Math.ceil(fullOriginalText.length / 500) * 10000),
            );
            const response = await fetchWithTimeout("/api/translate/transcript", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: fullOriginalText,
                    segments: originals,
                    source_lang: sourceLangSel.value,
                    target_lang: targetLangSel.value,
                    glossary: glossaryObject(),
                }),
            }, timeoutMs);
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.message || data.error || `HTTP ${response.status}`);
            }
            fullTextResult = data;
            renderFullTextResult(data);
            flashHint(fullTextWarningText(data) || "全文翻译已生成");
        } catch (err) {
            fullTextStatus.textContent = "生成失败";
            fullTextBody.innerHTML = `<div class="fulltext-empty">生成失败：${escHtml(err.message || err)}</div>`;
            flashHint("全文翻译生成失败：" + (err.message || err));
            failed = true;
        } finally {
            isFullTextTranslating = false;
            setHint(isRecording ? (captureMode === "system" ? "正在监听系统音频..." : "正在监听麦克风...") : idleHintText());
            if (!failed) renderFullTextResult(fullTextResult);
            updateControls();
        }
    }

    async function copyFullTextComparison() {
        const text = buildFullTextComparison();
        if (!text) {
            flashHint("当前没有可复制的全文翻译");
            return;
        }
        try {
            await navigator.clipboard.writeText(text);
            flashHint("全文翻译已复制");
        } catch {
            flashHint("复制失败，浏览器拒绝了剪贴板权限");
        }
    }

    function exportFullTextComparison() {
        const text = buildFullTextComparison();
        if (!text) {
            flashHint("当前没有可导出的全文翻译");
            return;
        }
        downloadBlob(text, `translive-fulltext-${Date.now()}.txt`, "text/plain;charset=utf-8");
        flashHint("全文翻译已导出");
    }

    function clearFullTextResult() {
        fullTextResult = null;
        isFullTextTranslating = false;
        renderFullTextResult(null);
        updateControls();
    }

    function savedRecordText(record) {
        const lines = [];
        const ts = fmtHistoryDate(record);
        const src = langLabel(record.source_lang);
        const tgt = langLabel(record.target_lang);
        if (record.original_text) lines.push(`[${ts}] (${src}) ${record.original_text}`);
        if (record.translated_text) lines.push(`[${ts}] (${tgt}) ${record.translated_text}`);
        return lines.join("\n").trim();
    }

    function savedRecordsText(records) {
        return (records || [])
            .map(savedRecordText)
            .filter(Boolean)
            .join("\n\n")
            .trim();
    }

    async function copySavedRecord(record) {
        const text = savedRecordText(record);
        if (!text) {
            flashHint("这条历史记录没有可复制内容");
            return;
        }
        try {
            await navigator.clipboard.writeText(text);
            flashHint("已复制这条历史记录");
        } catch {
            flashHint("复制失败，浏览器拒绝了剪贴板权限");
        }
    }

    async function copyFilteredSavedRecords() {
        const records = visibleSavedTranscriptRecords.length ? visibleSavedTranscriptRecords : filteredSavedRecords();
        const text = savedRecordsText(records);
        if (!text) {
            flashHint("没有可复制的筛选历史");
            return;
        }
        try {
            await navigator.clipboard.writeText(text);
            flashHint(`已复制 ${records.length} 条筛选历史`);
        } catch {
            flashHint("复制失败，浏览器拒绝了剪贴板权限");
        }
    }

    function exportFilteredSavedRecords() {
        const records = visibleSavedTranscriptRecords.length ? visibleSavedTranscriptRecords : filteredSavedRecords();
        const text = savedRecordsText(records);
        if (!text) {
            flashHint("没有可导出的筛选历史");
            return;
        }
        downloadBlob(text, `translive-history-${Date.now()}.txt`, "text/plain;charset=utf-8");
        flashHint(`已导出 ${records.length} 条筛选历史`);
    }

    function resetTranscriptView(options = {}) {
        const {
            notifySubtitle = true,
            clearStoredHistory = true,
            showFlash = false,
        } = options;
        clearTimeout(historySaveTimer);
        historySaveTimer = 0;
        segments.clear();
        clearFullTextResult();
        if (clearStoredHistory) localStorage.removeItem(HISTORY_KEY);
        transcript.innerHTML = "";
        transcript.appendChild(emptyState);
        userScrolledAwayFromLatest = false;
        jumpPill.hidden = true;
        updateControls();
        return notifySubtitle ? clearSubtitleWindow({ showFlash }) : Promise.resolve(true);
    }

    async function clearTranscript() {
        await resetTranscriptView({ notifySubtitle: true, showFlash: true });
    }

    function applyRecordToTranscript(record) {
        const entry = ensureCard(++lastSegId, record.source_lang || sourceLangSel.value, record.target_lang || targetLangSel.value);
        const ts = historyRecordDate(record);
        entry.ts = ts;
        entry.tsEl.textContent = fmtTime(ts);
        entry.tsEl.dateTime = ts.toISOString();
        entry.originalText = record.original_text || "";
        entry.translatedText = record.translated_text || "";
        entry.original.textContent = entry.originalText;
        entry.translated.textContent = entry.translatedText;
        entry.audioMs = record.audio_duration_ms ?? null;
        entry.asrMs = record.asr_ms ?? null;
        entry.mtMs = record.mt_ms ?? null;
        entry.totalMs = record.total_ms ?? null;
        entry.rtf = record.rtf ?? null;
        entry.warning = record.warning || "";
        entry.truncated = !!record.truncated;
        entry.translatedRow.classList.remove("pending", "streaming", "error");
        if (record.state === "error") entry.translatedRow.classList.add("error");
        if (!entry.translatedText) entry.translatedRow.classList.add("pending");
        setMetrics(entry);
    }

    function loadOneSavedRecordIntoTranscript(record) {
        if (isRecording || isStarting) {
            flashHint("同传运行中，停止后再载入历史记录");
            return;
        }
        applyRecordToTranscript(record);
        trimSegments();
        saveHistory();
        userScrolledAwayFromLatest = false;
        scrollToLatest();
        flashHint("已载入这条历史记录");
    }

    function loadSavedRecordsIntoTranscript() {
        if (isRecording || isStarting) {
            flashHint("同传运行中，停止后再载入历史记录");
            return;
        }
        const recordsToLoad = visibleSavedTranscriptRecords.length ? visibleSavedTranscriptRecords : filteredSavedRecords();
        if (!recordsToLoad.length) {
            flashHint("暂无历史记录可载入");
            return;
        }
        resetTranscriptView({ notifySubtitle: false, clearStoredHistory: true });
        isRestoringHistory = true;
        try {
            recordsToLoad.forEach(applyRecordToTranscript);
        } finally {
            isRestoringHistory = false;
        }
        trimSegments();
        saveHistory();
        userScrolledAwayFromLatest = false;
        scrollToLatest();
        flashHint(`已载入 ${recordsToLoad.length} 条历史记录`);
    }

    async function handleHistoryItemAction(event) {
        const target = event.target instanceof Element ? event.target : null;
        const button = target?.closest("[data-history-action]");
        if (!button || !historyList?.contains(button)) return;
        const item = button.closest("[data-history-index]");
        const index = Number(item?.dataset?.historyIndex);
        if (!Number.isInteger(index)) return;
        const record = visibleSavedTranscriptRecords[index];
        if (!record) return;
        const action = button.dataset.historyAction;
        if (action === "load") {
            loadOneSavedRecordIntoTranscript(record);
        } else if (action === "copy") {
            await copySavedRecord(record);
        }
    }

    function handleLanguageChange() {
        savePrefs();
        updateSwapAvailability();
        if (captureMode === "system" && isRecording) {
            sendSystemAudioConfig();
        } else {
            sendConfig();
        }
        if (sourceLangSel.value === "auto") {
            flashHint("源语言已设为自动检测，识别到什么语言就按什么语言翻译");
        } else if (sourceLangSel.value === targetLangSel.value) {
            flashHint("源语言和目标语言相同，将直接显示原文");
        }
    }

    function handleLatencyModeChange() {
        savePrefs();
        if (captureMode === "system" && isRecording) {
            sendSystemAudioConfig();
        } else {
            sendConfig();
        }
        flashHint(`识别模式：${latencyModeLabel(latencyModeSel.value)}`);
    }

    function handleStreamingPreviewChange() {
        savePrefs();
        if (captureMode === "system" && isRecording) {
            sendSystemAudioConfig();
        } else {
            sendConfig();
        }
        flashHint(streamingPreviewChk.checked ? "流式预览已开启" : "流式预览已关闭");
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

    historyToggle.addEventListener("click", () => {
        const willOpen = historyPanel.hidden;
        historyPanel.hidden = !willOpen;
        historyToggle.classList.toggle("active", willOpen);
        if (willOpen && savedTranscriptRecords.length === 0 && !isHistoryLoading) {
            fetchRecentTranscripts(false);
        }
    });

    historyRefreshBtn.addEventListener("click", () => fetchRecentTranscripts(true));
    if (historyCopyFilteredBtn) historyCopyFilteredBtn.addEventListener("click", copyFilteredSavedRecords);
    if (historyExportFilteredBtn) historyExportFilteredBtn.addEventListener("click", exportFilteredSavedRecords);
    historyLoadBtn.addEventListener("click", loadSavedRecordsIntoTranscript);
    historyClearSavedBtn.addEventListener("click", clearSavedTranscripts);
    historySearch.addEventListener("input", renderSavedRecords);
    historyStateFilter.addEventListener("change", renderSavedRecords);
    historyList.addEventListener("click", handleHistoryItemAction);

    if (fullTextToggle) {
        fullTextToggle.addEventListener("click", () => {
            const willOpen = fullTextPanel.hidden;
            fullTextPanel.hidden = !willOpen;
            fullTextToggle.classList.toggle("active", willOpen);
            renderFullTextResult(fullTextResult);
        });
    }
    if (fullTextGenerateBtn) fullTextGenerateBtn.addEventListener("click", generateFullTextTranslation);
    if (fullTextCopyBtn) fullTextCopyBtn.addEventListener("click", copyFullTextComparison);
    if (fullTextExportBtn) fullTextExportBtn.addEventListener("click", exportFullTextComparison);
    if (fullTextCloseBtn) {
        fullTextCloseBtn.addEventListener("click", () => {
            fullTextPanel.hidden = true;
            fullTextToggle?.classList.remove("active");
        });
    }

    // ── 事件绑定 ────────────────────────────────────────

    document.addEventListener("pointerdown", markUserActivity, { passive: true });
    document.addEventListener("keydown", markUserActivity);

    toggleBtn.addEventListener("click", async () => {
        isRecording ? await stopRecording() : await startRecording();
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
        if (audioSourceSel.value === "system") {
            if (nativeAudioAvailable) {
                setAudioSourceStatus("可用", "ok", audioSourceSel.title || "系统音频可用");
            } else {
                setAudioSourceStatus("不可用", "warn", audioSourceSel.title || "系统音频不可用");
                flashHint(audioSourceSel.title || "系统音频不可用，已使用麦克风");
            }
        } else {
            setAudioSourceStatus(nativeAudioAvailable ? "系统可用" : "麦克风", nativeAudioAvailable ? "ok" : "warn",
                nativeAudioAvailable ? "可随时切换到系统音频" : "当前使用麦克风输入");
        }
        setHint(idleHintText());
    });
    latencyModeSel.addEventListener("change", handleLatencyModeChange);
    streamingPreviewChk.addEventListener("change", handleStreamingPreviewChange);
    autoScrollChk.addEventListener("change", savePrefs);
    if (autoUnloadChk) {
        autoUnloadChk.addEventListener("change", () => {
            savePrefs();
            if (autoUnloadEnabled()) {
                lastUserActivityAt = Date.now();
                scheduleAutoUnload();
                flashHint("空闲释放已开启");
            } else {
                clearAutoUnloadTimer();
                flashHint("空闲释放已关闭");
            }
            updateControls();
        });
    }
    persistHistoryChk.addEventListener("change", () => {
        savePrefs();
        if (captureMode === "system" && isRecording) {
            sendSystemAudioConfig();
        } else {
            sendConfig();
        }
        if (historyEnabled()) {
            saveHistory();
            flashHint("保存记录已开启");
        }
        else {
            clearTimeout(historySaveTimer);
            historySaveTimer = 0;
            localStorage.removeItem(HISTORY_KEY);
            renderHistoryInfo();
            flashHint("保存记录已关闭，本轮不会继续落盘");
        }
    });
    clearBtn.addEventListener("click", clearTranscript);
    copyBtn.addEventListener("click", copyAll);
    exportTxtBtn.addEventListener("click", () =>
        downloadBlob(buildPlain(), `translive-${Date.now()}.txt`, "text/plain;charset=utf-8"));
    exportSrtBtn.addEventListener("click", () =>
        downloadBlob(buildSrt(), `translive-${Date.now()}.srt`, "text/plain;charset=utf-8"));
    modelDownloadBtn.addEventListener("click", downloadModels);
    if (modelUnloadBtn) modelUnloadBtn.addEventListener("click", unloadModels);
    subtitleModeSel.addEventListener("change", () => setSubtitleMode(subtitleModeSel.value));
    subtitleWindowBtn.addEventListener("click", openSubtitleWindow);
    if (clickThroughToggle) clickThroughToggle.addEventListener("change", applyClickThrough);
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
            setAudioSourceStatus("检测中", "warn", "正在等待桌面壳注入系统音频接口");
            audioSourceSel.title = "正在检测系统音频...";
            updateControls();
            if (nativeAudioInitAttempts < 30) {
                nativeAudioInitAttempts += 1;
                scheduleNativeAudioInit();
                return;
            }
            audioSourceSel.title = "系统音频仅支持 macOS 桌面版";
            setAudioSourceStatus("仅桌面版", "warn", audioSourceSel.title);
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
        setAudioSourceStatus("检测中", "warn", "正在检测 ScreenCaptureKit 系统音频能力");
        try {
            const result = await desktopCall("native_audio_available");
            nativeAudioAvailable = !!result?.ok;
            audioSourceSel.title = result?.reason || "音频输入";
            const needsScreenPermission = result?.diagnostics?.screen_recording_permission === false;
            setAudioSourceStatus(
                nativeAudioAvailable ? (needsScreenPermission ? "需授权" : "可用") : "不可用",
                nativeAudioAvailable ? (needsScreenPermission ? "warn" : "ok") : "warn",
                audioSourceSel.title,
            );
        } catch (err) {
            nativeAudioAvailable = false;
            audioSourceSel.title = err?.message || "系统音频检测失败";
            setAudioSourceStatus("检测失败", "error", audioSourceSel.title);
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

    async function refreshNativeAudioStatus() {
        if (!window.pywebview?.api?.native_audio_status) return;
        try {
            const result = await desktopCall("native_audio_status");
            if (result?.running) {
                const dropped = Number(result.dropped_chunks || 0);
                setAudioSourceStatus(
                    dropped > 0 ? "采集中*" : "采集中",
                    dropped > 0 ? "warn" : "ok",
                    dropped > 0
                        ? `系统音频正在采集；已丢弃 ${dropped} 个积压音频分片`
                        : "系统音频正在采集",
                );
            } else if (result?.error) {
                setAudioSourceStatus("错误", "error", result.error);
            } else if (nativeAudioAvailable) {
                setAudioSourceStatus("可用", "ok", audioSourceSel.title || "系统音频可用");
            }
        } catch {
            // 保留上一次可见状态即可。
        }
    }

    window.addEventListener("offline", () => {
        scheduleHealthPoll(HEALTH_POLL_ERROR_MS);
        if (isRecording || reconnectTimer) {
            setStatus("离线", "error");
            setHint("网络已离线，等待恢复连接...");
        }
    });

    window.addEventListener("online", () => {
        runScheduledHealthPoll();
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = 0;
            scheduleReconnect({ resumeMode: isRecording || captureMode === "mic" ? "mic" : null });
        } else if (!isRecording && backendReady && !ws) {
            setHint(idleHintText());
        }
        if (captureMode === "system" && isRecording && !subtitleMirrorWs) {
            scheduleSubtitleMirrorReconnect();
        }
    });

    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            scheduleHealthPoll(HEALTH_POLL_HIDDEN_MS);
        } else {
            runScheduledHealthPoll();
        }
    });

    window.addEventListener("beforeunload", () => {
        if (historySaveTimer) saveHistory();
        clearHealthPollTimer();
        clearAutoUnloadTimer();
        clearReconnect();
        clearSubtitleMirrorReconnect();
        if (captureMode === "system") {
            try { desktopCall("stop_system_audio_capture"); } catch {}
        }
        if (ws) {
            try { ws.close(); } catch {}
            ws = null;
        }
        if (subtitleMirrorWs) {
            try { subtitleMirrorWs.close(); } catch {}
            subtitleMirrorWs = null;
        }
        cleanupAudio();
    });

    loadPrefs();
    restoreHistory();
    updateControls();
    window.addEventListener("pywebviewready", () => {
        nativeAudioInitAttempts = 0;
        scheduleNativeAudioInit(0);
    });
    initNativeAudioOption();
    runScheduledHealthPoll();
})();

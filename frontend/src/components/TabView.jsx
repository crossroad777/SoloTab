import React, { useEffect, useRef, useState } from "react";
// ScoreToolbar: /score API未実装のため一時無効化 (コンポーネントはファイルとして残存)
// import ScoreToolbar from "./ScoreToolbar";

/**
 * TabView — AlphaTab TAB 譜表示
 * - カスタムBeatMapでtick→座標マッピング
 * - カスタム青カーソルバー + オートスクロール
 */
const TabViewInner = ({ sessionId, apiBase, currentTime, isPlaying, transpose = 0, capo = 0, onApiReady }) => {
    const containerRef = useRef(null);
    const wrapperRef = useRef(null);
    const cursorRef = useRef(null);
    const apiRef = useRef(null);
    const beatMapRef = useRef([]);
    const boundsReadyRef = useRef(false);
    const timeRef = useRef(0);
    const playingRef = useRef(false);
    const initKeyRef = useRef(null);

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [autoScroll, setAutoScroll] = useState(true);
    const autoScrollRef = useRef(true);
    const [scale, setScale] = useState(0.75);
    const scaleRef = useRef(0.75);



    // --- TAB編集UI state ---
    const [editNote, setEditNote] = useState(null); // {noteIndex, fret, string, x, y}
    const [editInput, setEditInput] = useState("");
    const [editSaving, setEditSaving] = useState(false);
    const [reloadKey, setReloadKey] = useState(0);
    const editInputRef = useRef(null);


    useEffect(() => {
        timeRef.current = currentTime;
        playingRef.current = isPlaying;
    }, [currentTime, isPlaying]);

    // ============================================================
    // BeatMap: 小節単位 — beats.jsonの実時刻 + AlphaTab bar座標
    // ============================================================
    const beatsDataRef = useRef([]); // beats.json の beat timestamps(秒)

    const buildBeatMap = (api) => {
        if (!api.score || !api.renderer?.boundsLookup) {
            console.log(`[TabView] BeatMap: score=${!!api.score}, boundsLookup=${!!api.renderer?.boundsLookup}`);
            return false;
        }
        const lookup = api.renderer.boundsLookup;
        // AlphaTab 1.x: staffSystems / staveGroups — バージョンにより名前が異なる
        const systems = lookup.staffSystems || lookup.staveGroups;
        if (!systems || systems.length === 0) {
            // boundsLookupの全プロパティ名をログ出力してデバッグ
            const keys = Object.keys(lookup).filter(k => typeof lookup[k] !== 'function');
            console.warn(`[TabView] BeatMap: no staffSystems/staveGroups. Available keys:`, keys);
            return false;
        }
        const beats = beatsDataRef.current;

        // AlphaTabから小節の座標を取得
        // Y/高さはスタッフシステム（1段）の境界を使用し、X/幅は個別小節から取得
        // → エフェクト記号(let ring, vibrato等)による小節ごとのY変動を排除
        const barCoords = [];
        let barCounterFallback = 0;
        for (const system of systems) {
            const sgBars = system.bars || system.masterBars;
            if (!sgBars) continue;
            const sysVb = system.visualBounds;
            for (const sgBar of sgBars) {
                const barIndex = sgBar.index ?? sgBar.barIndex ?? barCounterFallback;
                barCounterFallback++;
                const barVb = sgBar.visualBounds;
                if (barVb == null || barVb.x == null) continue;
                barCoords.push({
                    barIndex,
                    vb: {
                        x: barVb.x,
                        y: sysVb ? sysVb.y : barVb.y,
                        w: barVb.w,
                        h: sysVb ? sysVb.h : barVb.h,
                    },
                });
            }
        }
        barCoords.sort((a, b) => a.barIndex - b.barIndex);
        // Deduplicate: ScoreTab mode has 2 staves (notation + TAB) per bar
        // → same barIndex appears twice. Keep only the first (wider coverage).
        const seen = new Set();
        const uniqueBarCoords = [];
        for (const bc of barCoords) {
            if (!seen.has(bc.barIndex)) {
                seen.add(bc.barIndex);
                uniqueBarCoords.push(bc);
            }
        }
        const finalBarCoords = uniqueBarCoords;
        console.log(`[TabView] barCoords: ${barCoords.length} raw → ${finalBarCoords.length} deduplicated`);
        if (finalBarCoords.length === 0) {
            console.warn(`[TabView] BeatMap: barCoords empty. systems=${systems.length}`);
            return false;
        }

        // 拍子を取得
        const masterBars = api.score.masterBars;
        const getBeatsPerBar = (idx) => {
            let arr = [];
            const forEach = (list, cb) => {
                if (Array.isArray(list)) list.forEach(cb);
                else if (list?.items) list.items.forEach(cb);
                else if (list?.forEach) list.forEach(cb);
            };
            if (masterBars) forEach(masterBars, mb => arr.push(mb));
            return idx < arr.length ? (arr[idx].timeSignatureNumerator || 4) : 4;
        };

        // beats.jsonのタイムスタンプを使って各小節の開始/終了時刻を計算
        const map = [];
        let beatIdx = 0;
        for (const bc of finalBarCoords) {
            const bpb = getBeatsPerBar(bc.barIndex);
            const startMs = beatIdx < beats.length ? beats[beatIdx] * 1000 : null;
            const endBeatIdx = beatIdx + bpb;
            const endMs = endBeatIdx < beats.length ? beats[endBeatIdx] * 1000 : null;

            if (startMs == null) break;

            map.push({
                startMs,
                endMs: endMs ?? (startMs + bpb * 600), // fallback
                vb: bc.vb,
            });
            beatIdx += bpb;
        }

        if (map.length === 0) {
            // beats.jsonがない場合はテンポから計算（フォールバック）
            const bpm = api.score.tempo || 120;
            let accMs = 0;
            for (const bc of finalBarCoords) {
                const bpb = getBeatsPerBar(bc.barIndex);
                const durMs = bpb * (60000 / bpm);
                map.push({ startMs: accMs, endMs: accMs + durMs, vb: bc.vb });
                accMs += durMs;
            }
        }

        beatMapRef.current = map;
        console.log(`[TabView] BarMap: ${map.length} bars, first=${(map[0].startMs / 1000).toFixed(2)}s`);
        return true;
    };

    const findBeat = (audioMs) => {
        const map = beatMapRef.current;
        if (!map || !map.length) return null;
        if (audioMs < map[0].startMs) return map[0];
        if (audioMs >= map[map.length - 1].startMs) return map[map.length - 1];
        let lo = 0, hi = map.length - 1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (audioMs >= map[mid].startMs && audioMs < map[mid].endMs) return map[mid];
            if (audioMs < map[mid].startMs) hi = mid - 1;
            else lo = mid + 1;
        }
        return lo > 0 ? map[lo - 1] : map[0];
    };

    // ============================================================
    // AlphaTab init — NO innerHTML manipulation on React refs
    // ============================================================
    useEffect(() => {
        if (!sessionId || !wrapperRef.current) return;

        const key = `${sessionId}_${transpose}_${capo}_${reloadKey}`;
        if (initKeyRef.current === key) return;
        initKeyRef.current = key;

        let destroyed = false;
        boundsReadyRef.current = false;
        beatMapRef.current = [];

        // Destroy old API
        if (apiRef.current) {
            try { apiRef.current.destroy(); } catch { /* noop */ }
            apiRef.current = null;
        }
        // Clear only the AlphaTab-generated content inside wrapper
        while (wrapperRef.current.firstChild) {
            wrapperRef.current.removeChild(wrapperRef.current.firstChild);
        }

        const init = async () => {
            setLoading(true);
            setError(null);

            try {
                // GP5バイナリを取得
                let res;
                let useGp5 = true;
                for (let attempt = 0; attempt < 3; attempt++) {
                    res = await fetch(`${apiBase}/result/${sessionId}/gp5?t=${Date.now()}`);
                    if (res.ok) break;
                    console.warn(`[TabView] GP5 attempt ${attempt + 1} failed, retrying...`);
                    await new Promise(r => setTimeout(r, 1500));
                }
                if (!res || !res.ok) {
                    console.warn("[TabView] GP5 not available, falling back to MusicXML");
                    useGp5 = false;
                    for (let attempt = 0; attempt < 3; attempt++) {
                        res = await fetch(`${apiBase}/result/${sessionId}/musicxml?t=${Date.now()}`);
                        if (res.ok) break;
                        await new Promise(r => setTimeout(r, 1500));
                    }
                    if (!res.ok) throw new Error("Score not available");
                }

                const scoreData = useGp5
                    ? new Uint8Array(await res.arrayBuffer())
                    : new TextEncoder().encode(await res.text());
                if (destroyed) return;

                console.log(`[TabView] Loading ${useGp5 ? 'GP5' : 'MusicXML'}: ${scoreData.length} bytes`);

                // beats.jsonを取得（カーソル同期用）
                try {
                    const beatRes = await fetch(`${apiBase}/result/${sessionId}/beats`);
                    if (beatRes.ok) {
                        const beatData = await beatRes.json();
                        beatsDataRef.current = beatData.beats || [];
                        console.log(`[TabView] Loaded ${beatsDataRef.current.length} beats for cursor sync`);
                    }
                } catch { /* ignore */ }

                // タイトル取得（GP5のLatin-1制限回避）
                let songTitle = null;
                try {
                    const infoRes = await fetch(`${apiBase}/result/${sessionId}`);
                    if (infoRes.ok) {
                        const info = await infoRes.json();
                        songTitle = info.filename || null;
                        if (songTitle) {
                            const audioExts = ['.mp3','.wav','.m4a','.flac','.ogg','.opus','.webm','.mp4'];
                            for (const ext of audioExts) {
                                if (songTitle.toLowerCase().endsWith(ext)) {
                                    songTitle = songTitle.slice(0, -ext.length);
                                    break;
                                }
                            }
                            // Remove junk metadata patterns from filename
                            songTitle = songTitle
                                .replace(/\s*\(\d+k\)/gi, '')       // (128k)
                                .replace(/\s*Tab譜.*$/i, '')         // Tab譜 楽譜 ...
                                .replace(/\s*ギター\s*タブ.*$/i, '') // ギター タブ ...
                                .replace(/\s*コードネーム付\s*/gi, '') // コードネーム付
                                .replace(/\s*-\s*アコースティック.*$/i, '') // - アコースティック ...
                                .replace(/\s*楽譜.*$/i, '')          // 楽譜...
                                .trim();
                        }
                    }
                } catch { /* ignore */ }

                if (!window.alphaTab) throw new Error("AlphaTab not loaded");

                const settings = new window.alphaTab.Settings();
                settings.core.tex = false;
                settings.core.fontDirectory = "https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/font/";

                // === 五線譜 + TAB 2段表示 ===
                settings.display.staveProfile = window.alphaTab.StaveProfile.ScoreTab;
                settings.display.layoutMode = window.alphaTab.LayoutMode.Page;
                settings.display.scale = scaleRef.current;
                settings.display.stretchForce = 1.2;
                settings.display.barsPerRow = 4;

                // === 記譜設定 ===
                settings.notation.rhythmMode = 0;
                settings.notation.fingeringMode = 0;

                // === タイトル非表示（GP5のLatin-1エンコードで文字化けするため） ===
                const NE = window.alphaTab.NotationElement;
                if (NE) {
                    settings.notation.elements.set(NE.ScoreTitle, false);
                    settings.notation.elements.set(NE.ScoreSubTitle, false);
                    settings.notation.elements.set(NE.ScoreArtist, false);
                    settings.notation.elements.set(NE.ScoreWordsAndMusic, false);
                }

                if (settings.display.resources) {
                    settings.display.resources.titleFont = new window.alphaTab.model.Font("Arial", 16, 1);
                }

                // === Player: boundsLookup生成に必要なので有効化、カーソルは無効 ===
                settings.player.enablePlayer = true;
                settings.player.enableCursor = true;
                settings.player.scrollMode = 0;
                settings.player.soundFont = "https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/soundfont/sonivox.sf2";
                settings.core.includeNoteBounds = true;

                const api = new window.alphaTab.AlphaTabApi(wrapperRef.current, settings);
                apiRef.current = api;
                if (onApiReady) {
                    onApiReady(api);
                }

                // タイトル上書き: 描画前にGP5の文字化けタイトルを消す
                api.scoreLoaded.on((score) => {
                    if (score) {
                        score.title = songTitle || '';
                        score.subTitle = '';
                        score.artist = songTitle ? 'SoloTab' : '';
                        score.words = '';
                        score.music = '';
                    }
                });

                // --- ノートクリック → 編集UI ---
                api.noteMouseDown.on((note, evt) => {
                    if (!note || !containerRef.current) return;
                    let noteIdx = 0;
                    let found = false;
                    const score = api.score;
                    if (!score) return;
                    for (const track of score.tracks) {
                        for (const staff of track.staves) {
                            for (const bar of staff.bars) {
                                for (const voice of bar.voices) {
                                    for (const beat of voice.beats) {
                                        if (beat.isRest) continue;
                                        for (const n of beat.notes) {
                                            if (n === note) { found = true; break; }
                                            noteIdx++;
                                        }
                                        if (found) break;
                                    }
                                    if (found) break;
                                }
                                if (found) break;
                            }
                            if (found) break;
                        }
                        if (found) break;
                    }
                    if (!found) return;

                    const rect = containerRef.current.getBoundingClientRect();
                    let px, py;
                    if (evt && (evt.pageX || evt.clientX)) {
                        px = (evt.pageX || evt.clientX) - rect.left;
                        py = (evt.pageY || evt.clientY) - rect.top + containerRef.current.scrollTop;
                    } else {
                        // AlphaTab 1.3.0: evt is undefined, use boundsLookup for note position
                        const bl = api.renderer?.boundsLookup;
                        let noteBounds = null;
                        if (bl) {
                            try {
                                // Try to find note bounds through boundsLookup
                                const groups = bl.staffSystems || bl.staveGroups || [];
                                outer: for (const sys of groups) {
                                    const bars = sys.bars || sys.masterBars || [];
                                    for (const bar of bars) {
                                        const barBounds = bar.bars || [];
                                        for (const bb of barBounds) {
                                            const beats = bb.beats || [];
                                            for (const beatBounds of beats) {
                                                const notes = beatBounds.notes || [];
                                                for (const nb of notes) {
                                                    if (nb.note === note) {
                                                        noteBounds = nb.noteHeadBounds || beatBounds.visualBounds;
                                                        break outer;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } catch { /* ignore */ }
                        }
                        if (noteBounds) {
                            px = noteBounds.x + noteBounds.w / 2;
                            py = noteBounds.y;
                        } else {
                            // Last resort: center of visible area
                            px = rect.width / 2;
                            py = containerRef.current.scrollTop + rect.height / 3;
                        }
                    }

                    setEditNote({ noteIndex: noteIdx, fret: note.fret, string: note.string, x: px, y: py, alphaNote: note });
                    setEditInput(String(note.fret));
                    setTimeout(() => editInputRef.current?.focus(), 50);
                });

                api.renderStarted.on(() => setLoading(true));
                api.postRenderFinished.on(() => {
                    if (destroyed) return;
                    setLoading(false);


                    // Build BeatMap with retries
                    const tryBuild = (attempt) => {
                        if (destroyed || boundsReadyRef.current) return;
                        const ok = buildBeatMap(api);
                        boundsReadyRef.current = ok;
                        if (ok) console.log("[TabView] BeatMap ready");
                        else if (attempt < 4) {
                            setTimeout(() => tryBuild(attempt + 1), [500, 1000, 2000, 3000][attempt]);
                        }
                    };
                    tryBuild(0);
                    if (containerRef.current) containerRef.current.scrollTop = 0;
                });
                api.error.on((e) => {
                    console.error("[AlphaTab Error]", e);
                    if (!destroyed) { setError("TAB表示エラー"); setLoading(false); }
                });

                api.load(scoreData);
            } catch (err) {
                console.error("[TabView init]", err);
                if (!destroyed) { setError(err.message); setLoading(false); }
            }
        };

        init();
        return () => {
            destroyed = true;
            boundsReadyRef.current = false;
            initKeyRef.current = null;
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sessionId, apiBase, transpose, capo, reloadKey]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (apiRef.current) {
                try { apiRef.current.destroy(); } catch { /* noop */ }
                apiRef.current = null;
            }
        };
    }, []);

    // ============================================================
    // Sync loop: cursor + auto-scroll
    // ============================================================
    useEffect(() => {
        let animId;
        let lastScrollMs = 0;
        let wasPlaying = false;

        const sync = () => {
            const cursor = cursorRef.current;
            const container = containerRef.current;
            const ms = Math.max(0, timeRef.current * 1000);
            const nowPlaying = playingRef.current;

            // 再生開始時にスクロールリセット
            if (nowPlaying && !wasPlaying && container && ms < 1000) {
                container.scrollTo({ top: 0, behavior: "instant" });
            }
            wasPlaying = nowPlaying;

            if (cursor && boundsReadyRef.current) {
                // 再生中のみカーソル表示
                if (!nowPlaying) {
                    cursor.style.display = "none";
                    animId = requestAnimationFrame(sync);
                    return;
                }

                const beat = findBeat(ms);
                if (beat) {
                    const { x, y, w, h } = beat.vb;

                    // 小節全体をハイライト（小節ごとに移動）
                    cursor.style.display = "block";
                    cursor.style.left = `${x}px`;
                    cursor.style.top = `${y}px`;
                    cursor.style.width = `${w}px`;
                    cursor.style.height = `${h}px`;

                    if (container && autoScrollRef.current) {
                        const now = Date.now();
                        if (now - lastScrollMs > 400) {
                            const cursorScreenY = y - container.scrollTop;
                            const viewH = container.clientHeight;
                            if (cursorScreenY < 0 || cursorScreenY > viewH * 0.55) {
                                container.scrollTo({ top: Math.max(0, y - viewH * 0.3), behavior: "smooth" });
                                lastScrollMs = now;
                            }
                        }
                    }
                } else {
                    cursor.style.display = "none";
                }
            }
            animId = requestAnimationFrame(sync);
        };

        animId = requestAnimationFrame(sync);
        return () => cancelAnimationFrame(animId);
    }, []);

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
            {/* ScoreToolbar: /score API未実装のため一時無効化 */}
            <div
                ref={containerRef}
                className="tab-print-container"
                style={{
                    width: "100%", flex: 1,
                    overflow: "auto", position: "relative",
                    background: "white", paddingBottom: 500,
                }}
            >
            {loading && (
                <div style={{
                    position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
                    background: "rgba(255,255,255,0.95)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    zIndex: 40,
                }}>
                    <div style={{ textAlign: "center" }}>
                        <div style={{
                            width: 48, height: 48, margin: "0 auto 16px",
                            border: "4px solid #1a1a2e", borderTopColor: "transparent",
                            borderRadius: "50%", animation: "spin 1s linear infinite",
                        }} />
                        <div style={{ fontSize: 14, fontWeight: 800, color: "#1a1a2e", letterSpacing: 2 }}>
                            TAB譜を描画中...
                        </div>
                    </div>
                </div>
            )}
            {error && (
                <div style={{
                    padding: 40, textAlign: "center",
                    color: "#ef4444", fontSize: 14, fontWeight: 600,
                }}>
                    ❌ {error}
                </div>
            )}

            {/* Score container — position:relative for cursor positioning */}
            <div style={{ position: "relative", padding: 0, margin: 0 }}>
                {/* Custom cursor (blue bar) */}
                <div
                    ref={cursorRef}
                    style={{
                        position: "absolute", display: "none", pointerEvents: "none",
                        zIndex: 30, top: 0, left: 0,
                        background: "rgba(59,130,246,0.10)",
                        borderRadius: 4,
                        boxShadow: "none",
                        transition: "left 0.15s ease, top 0.05s ease, width 0.15s ease",
                        willChange: "left, top",
                    }}
                />
                {/* AlphaTab renders into this div */}
                <div ref={wrapperRef} className="alpha-tab-wrapper" style={{ width: "100%", minHeight: "100vh" }} />

                {/* ノート編集ポップアップ */}
                {editNote && (
                    <>
                    {/* 枠外クリックで閉じるオーバーレイ */}
                    <div
                        onClick={() => setEditNote(null)}
                        style={{
                            position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
                            zIndex: 99, background: "transparent",
                        }}
                    />
                    <div
                        style={{
                            position: "absolute",
                            left: Math.max(10, editNote.x - 80), top: editNote.y + 20,
                            zIndex: 100,
                            background: "rgba(20,20,30,0.97)",
                            border: "2px solid #3b82f6",
                            borderRadius: 12,
                            padding: "12px 14px",
                            boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
                            display: "flex", flexDirection: "column", gap: 8,
                            minWidth: 180,
                        }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600 }}>
                            🎸 弦{editNote.editString ?? editNote.string} フレット{editNote.fret}
                        </div>
                        {/* 弦選択 */}
                        <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
                            <span style={{ fontSize: 10, color: "#64748b", width: 20 }}>弦</span>
                            {[1,2,3,4,5,6].map(s => (
                                <button key={s}
                                    onClick={() => setEditNote(prev => ({ ...prev, editString: s }))}
                                    style={{
                                        width: 26, height: 26, borderRadius: 6, border: "none",
                                        background: (editNote.editString ?? editNote.string) === s ? "#3b82f6" : "#334155",
                                        color: "white", fontWeight: 700, fontSize: 12,
                                        cursor: "pointer", transition: "all 0.15s",
                                    }}
                                >{s}</button>
                            ))}
                        </div>
                        {/* フレット入力 + 保存 */}
                        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                            <span style={{ fontSize: 10, color: "#64748b", width: 20 }}>F</span>
                            <input
                                ref={editInputRef}
                                type="number"
                                min="0" max="15"
                                value={editInput}
                                onChange={(e) => setEditInput(e.target.value)}
                                onKeyDown={async (e) => {
                                    if (e.key === "Enter") {
                                        e.preventDefault();
                                        const newFret = parseInt(editInput);
                                        const newString = editNote.editString ?? editNote.string;
                                        if (isNaN(newFret) || newFret < 0 || newFret > 15) return;
                                        setEditSaving(true);
                                        try {
                                            await fetch(`${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`, {
                                                method: "PATCH",
                                                headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({ fret: newFret, string: newString }),
                                            });
                                            setEditNote(null);
                                            setReloadKey(k => k + 1);
                                        } catch (err) { console.error("Edit failed:", err); }
                                        setEditSaving(false);
                                    } else if (e.key === "Escape") {
                                        setEditNote(null);
                                    }
                                }}
                                style={{
                                    width: 50, padding: "4px 6px", borderRadius: 6,
                                    border: "1px solid #475569", background: "#1e293b",
                                    color: "white", fontSize: 16, fontWeight: 700,
                                    textAlign: "center", outline: "none",
                                }}
                                disabled={editSaving}
                            />
                            <button
                                onClick={async () => {
                                    const newFret = parseInt(editInput);
                                    const newString = editNote.editString ?? editNote.string;
                                    if (isNaN(newFret) || newFret < 0 || newFret > 15) return;
                                    setEditSaving(true);
                                    try {
                                        await fetch(`${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`, {
                                            method: "PATCH",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ fret: newFret, string: newString }),
                                        });
                                        setEditNote(null);
                                        setReloadKey(k => k + 1);
                                    } catch (err) { console.error("Edit failed:", err); }
                                    setEditSaving(false);
                                }}
                                disabled={editSaving}
                                style={{
                                    padding: "4px 10px", borderRadius: 6, border: "none",
                                    background: "#3b82f6", color: "white", fontWeight: 700,
                                    cursor: "pointer", fontSize: 13,
                                }}
                            >✓</button>
                            <button
                                onClick={async () => {
                                    if (!confirm("このノートを削除しますか？")) return;
                                    setEditSaving(true);
                                    try {
                                        await fetch(`${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`, {
                                            method: "PATCH",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ delete: true }),
                                        });
                                        setEditNote(null);
                                        setReloadKey(k => k + 1);
                                    } catch (err) { console.error("Delete failed:", err); }
                                    setEditSaving(false);
                                }}
                                disabled={editSaving}
                                style={{
                                    padding: "4px 8px", borderRadius: 6, border: "none",
                                    background: "#ef4444", color: "white", fontWeight: 700,
                                    cursor: "pointer", fontSize: 13,
                                }}
                            >🗑</button>
                        </div>
                        <div style={{ fontSize: 10, color: "#64748b" }}>枠外クリック or Esc=閉じる</div>
                    </div>
                    </>
                )}
            </div>

            {/* ズームコントロール + Auto-scroll */}
            <div style={{
                position: "fixed", bottom: 80, right: 24, zIndex: 50,
                display: "flex", gap: 8, alignItems: "center",
            }}>
                {/* ズームコントロール */}
                <div style={{
                    display: "flex", gap: 2, alignItems: "center",
                    background: "rgba(30,30,40,0.9)",
                    borderRadius: 20, padding: "4px 8px",
                    boxShadow: "0 4px 16px rgba(0,0,0,0.3)",
                }}>
                    <button
                        onClick={() => {
                            const next = Math.max(0.4, scale - 0.1);
                            setScale(next);
                            scaleRef.current = next;
                            if (apiRef.current) {
                                apiRef.current.settings.display.scale = next;
                                apiRef.current.updateSettings();
                                apiRef.current.render();
                            }
                        }}
                        style={{
                            width: 28, height: 28, borderRadius: "50%", border: "none",
                            background: "#334155", color: "white", fontSize: 16,
                            fontWeight: 700, cursor: "pointer", display: "flex",
                            alignItems: "center", justifyContent: "center",
                        }}
                    >−</button>
                    <span style={{
                        color: "white", fontSize: 11, fontWeight: 700,
                        minWidth: 38, textAlign: "center",
                    }}>{Math.round(scale * 100)}%</span>
                    <button
                        onClick={() => {
                            const next = Math.min(1.5, scale + 0.1);
                            setScale(next);
                            scaleRef.current = next;
                            if (apiRef.current) {
                                apiRef.current.settings.display.scale = next;
                                apiRef.current.updateSettings();
                                apiRef.current.render();
                            }
                        }}
                        style={{
                            width: 28, height: 28, borderRadius: "50%", border: "none",
                            background: "#334155", color: "white", fontSize: 16,
                            fontWeight: 700, cursor: "pointer", display: "flex",
                            alignItems: "center", justifyContent: "center",
                        }}
                    >+</button>
                </div>
                {/* Auto-scroll toggle */}
                <div
                    style={{
                        padding: "8px 16px", borderRadius: 20, cursor: "pointer",
                        background: autoScroll ? "#10b981" : "rgba(30,30,40,0.8)",
                        color: "white", fontSize: 12, fontWeight: 700,
                        boxShadow: "0 4px 16px rgba(0,0,0,0.3)",
                        transition: "all 0.2s", userSelect: "none",
                    }}
                    onClick={() => {
                        setAutoScroll((v) => {
                            const next = !v;
                            autoScrollRef.current = next;
                            return next;
                        });
                    }}
                >
                    {autoScroll ? "📌 AUTO SCROLL ON" : "✋ AUTO SCROLL OFF"}
                </div>
            </div>

            <style>{`
                .at-cursor-beat, .at-cursor-bar, .at-selection, .at-highlight { display: none !important; }
                .alphaTabSurface { position: static !important; }
                .at-surface { overflow: visible !important; }
                .at-main text { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
                .at-staff-tab .at-note-number { font-size: 13px !important; font-weight: 600 !important; }
                .at-effect-note text { font-size: 10px !important; fill: #555 !important; }
                .at-system { margin-bottom: 8px !important; }
                /* 三連符ブラケットと数字を非表示 */
                .at-tuplet-group { display: none !important; }
                svg g[data-name*="tuplet"] { display: none !important; }
                .at-score .at-tuplet { display: none !important; }
                @media print {
                    body { background: white !important; }
                    .at-surface { transform: scale(1) !important; width: 100% !important; overflow: visible !important; }
                    .at-system { break-inside: avoid !important; page-break-inside: avoid !important; margin-bottom: 16px !important; display: block !important; }
                    .at-viewport { overflow: visible !important; height: auto !important; display: block !important; }
                    .tab-print-container { height: auto !important; overflow: visible !important; padding-bottom: 0 !important; }
                }
            `}</style>
        </div>
        </div>
    );
};

// ErrorBoundary to prevent white screen
class TabViewErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false };
    }
    static getDerivedStateFromError() {
        return { hasError: true };
    }
    componentDidCatch(err) {
        console.error("[TabView] Caught error:", err);
    }
    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    display: "flex", flexDirection: "column", alignItems: "center",
                    justifyContent: "center", height: "60vh", color: "#f59e0b", gap: 16,
                }}>
                    <p style={{ fontSize: 18 }}>⚠️ 楽譜の描画中にエラーが発生しました</p>
                    <button
                        onClick={() => this.setState({ hasError: false })}
                        style={{
                            padding: "10px 24px", borderRadius: 8,
                            background: "#f59e0b", color: "#000", fontWeight: "bold",
                            border: "none", cursor: "pointer", fontSize: 14,
                        }}
                    >
                        再試行
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}

export function TabView(props) {
    return (
        <TabViewErrorBoundary>
            <TabViewInner {...props} />
        </TabViewErrorBoundary>
    );
}

export default TabView;

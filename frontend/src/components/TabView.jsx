import React, { useEffect, useRef, useState } from "react";

/**
 * TabView — AlphaTab TAB 譜表示
 * - enablePlayer: false (音声はApp.jsxのaudioタグで再生)
 * - カスタムBeatMapでtick→座標マッピング
 * - カスタム青カーソルバー + オートスクロール
 * - ErrorBoundaryでクラッシュ防止
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

    // --- TAB編集UI state ---
    const [editNote, setEditNote] = useState(null); // {noteIndex, fret, string, x, y}
    const [editInput, setEditInput] = useState("");
    const [editSaving, setEditSaving] = useState(false);
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
        if (!api.score || !api.renderer?.boundsLookup) return false;
        const lookup = api.renderer.boundsLookup;
        const systems = lookup.staffSystems;
        if (!systems || systems.length === 0) return false;
        const beats = beatsDataRef.current;

        // AlphaTabから小節の座標を取得
        const barCoords = [];
        for (const system of systems) {
            const sgBars = system.bars;
            if (!sgBars) continue;
            for (const sgBar of sgBars) {
                const barIndex = sgBar.index;
                const vb = sgBar.visualBounds;
                if (barIndex == null || !vb || vb.x == null) continue;
                barCoords.push({
                    barIndex,
                    vb: { x: vb.x, y: vb.y, w: vb.w, h: vb.h },
                });
            }
        }
        barCoords.sort((a, b) => a.barIndex - b.barIndex);
        if (barCoords.length === 0) return false;

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
        for (const bc of barCoords) {
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
            for (const bc of barCoords) {
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

        const key = `${sessionId}_${transpose}_${capo}`;
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
                // Cache-bust the fetch so new retunes are properly loaded instead of browser-cached versions
                const res = await fetch(`${apiBase}/result/${sessionId}/musicxml?t=${Date.now()}`);
                if (!res.ok) throw new Error("MusicXML not available");
                let xmlText = await res.text();
                if (destroyed) return;

                // beats.jsonを取得（カーソル同期用）
                try {
                    const beatRes = await fetch(`${apiBase}/result/${sessionId}/beats`);
                    if (beatRes.ok) {
                        const beatData = await beatRes.json();
                        beatsDataRef.current = beatData.beats || [];
                        console.log(`[TabView] Loaded ${beatsDataRef.current.length} beats for cursor sync`);
                    }
                } catch { /* ignore */ }
                if (!window.alphaTab) throw new Error("AlphaTab not loaded");

                // Transpose: modify MusicXML pitch directly
                if (transpose !== 0) {
                    const NOTES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"];
                    const ALTERS = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0];
                    const N2M = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
                    xmlText = xmlText.replace(
                        /<pitch>\s*<step>([A-G])<\/step>\s*(?:<alter>([-.\\d]+)<\/alter>\s*)?<octave>(\d+)<\/octave>\s*<\/pitch>/g,
                        (_, step, alter, octave) => {
                            const midi = N2M[step] + (alter ? parseInt(alter) : 0) + (parseInt(octave) + 1) * 12 + transpose;
                            const pc = ((midi % 12) + 12) % 12;
                            let r = `<pitch><step>${NOTES[pc]}</step>`;
                            if (ALTERS[pc]) r += `<alter>${ALTERS[pc]}</alter>`;
                            return r + `<octave>${Math.floor(midi / 12) - 1}</octave></pitch>`;
                        }
                    );
                    xmlText = xmlText.replace(
                        /<fret>(\d+)<\/fret>/g,
                        (_, f) => `<fret>${Math.max(0, parseInt(f) + transpose)}</fret>`
                    );
                }

                // Capo: バックエンド側でカポ対応tuningを使用して弦割り当て済み
                // フロントでのfret-capo処理は不要（二重補正回避）

                const settings = new window.alphaTab.Settings();
                settings.core.tex = false;
                settings.core.fontDirectory = "https://cdn.jsdelivr.net/npm/@coderline/alphatab@latest/dist/font/";

                // === TAB表示最適化 ===
                settings.display.staveProfile = window.alphaTab.StaveProfile.Tab;
                settings.display.layoutMode = window.alphaTab.LayoutMode.Page;
                settings.display.scale = 0.75;              // スケール縮小で密集解消
                settings.display.stretchForce = 1.2;        // ノート間スペースを広げる
                settings.display.barsPerRow = 4;             // 1行4小節で読みやすく

                // === 記譜設定 ===
                settings.notation.rhythmMode = window.alphaTab.TabRhythmMode?.ShowWithBars || 0;
                settings.notation.fingeringMode = 0;         // 運指表記OFF

                // テクニック記号の表示はAlphaTabのデフォルト設定に依存
                // MusicXMLのharmonic/staccato/bend/slide要素は自動検出される

                // タイトルフォントを小さくして長い曲名も収まるように
                if (settings.display.resources) {
                    settings.display.resources.titleFont = new window.alphaTab.model.Font("Arial", 16, 1); // 1=Bold
                }

                settings.player.enablePlayer = false;
                settings.player.enableCursor = false;
                settings.core.includeNoteBounds = true; // ノートクリック検出に必要

                const api = new window.alphaTab.AlphaTabApi(wrapperRef.current, settings);
                apiRef.current = api;
                if (onApiReady) {
                    onApiReady(api);
                }

                // --- ノートクリック → 編集UI ---
                api.noteMouseDown.on((note, evt) => {
                    if (!note || !containerRef.current) return;
                    // ノートインデックスを計算（score内の全ノートを走査）
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

                    // ポップアップ位置を計算
                    const rect = containerRef.current.getBoundingClientRect();
                    const px = (evt?.pageX || evt?.clientX || 200) - rect.left;
                    const py = (evt?.pageY || evt?.clientY || 200) - rect.top + containerRef.current.scrollTop;

                    setEditNote({ noteIndex: noteIdx, fret: note.fret, string: note.string, x: px, y: py, alphaNote: note });
                    setEditInput(String(note.fret));
                    setTimeout(() => editInputRef.current?.focus(), 50);
                });

                // テクニックマップをスコアモデルに適用する関数
                const applyTechniques = async (score) => {
                    try {
                        const techRes = await fetch(`${apiBase}/result/${sessionId}/techniques`);
                        if (!techRes.ok) return;
                        const techMap = await techRes.json();
                        if (!techMap || !techMap.length) return;

                        // スコアモデルのノートをMusicXML出力順に走査
                        let noteIdx = 0;
                        let applied = { palm_mute: 0, harmonic: 0, hammer_on: 0, pull_off: 0, slide: 0, bend: 0, vibrato: 0, let_ring: 0, ghost_note: 0, trill: 0 };
                        const HT = window.alphaTab.model?.HarmonicType;
                        const ST = window.alphaTab.model?.SlideOutType;
                        const BT = window.alphaTab.model?.BendType;
                        const VT = window.alphaTab.model?.VibratoType;

                        for (const track of score.tracks) {
                            for (const staff of track.staves) {
                                for (const bar of staff.bars) {
                                    for (const voice of bar.voices) {
                                        for (const beat of voice.beats) {
                                            if (beat.isRest) continue;
                                            for (const note of beat.notes) {
                                                if (noteIdx >= techMap.length) break;
                                                const tech = techMap[noteIdx];
                                                if (tech === "palm_mute") {
                                                    try {
                                                        note.isPalmMute = true;
                                                        applied.palm_mute++;
                                                    } catch { /* readonly fallback */ }
                                                } else if (tech === "harmonic" && HT) {
                                                    try {
                                                        note.harmonicType = HT.Natural;
                                                        applied.harmonic++;
                                                    } catch { /* getter-only */ }
                                                } else if (tech === "hammer_on") {
                                                    try {
                                                        note.isHammerPullOrigin = true;
                                                        applied.hammer_on++;
                                                    } catch { /* */ }
                                                } else if (tech === "pull_off") {
                                                    try {
                                                        note.isHammerPullOrigin = true;
                                                        applied.pull_off++;
                                                    } catch { /* */ }
                                                } else if ((tech === "slide_up" || tech === "slide_down") && ST) {
                                                    try {
                                                        note.slideOutType = tech === "slide_up" ? ST.Shift : ST.Shift;
                                                        applied.slide++;
                                                    } catch { /* */ }
                                                } else if (tech === "bend" && BT) {
                                                    try {
                                                        note.bendType = BT.Bend;
                                                        // デフォルトの1音ベンド (4 quarter tone steps = 1 whole tone)
                                                        if (note.bendPoints && note.bendPoints.length === 0) {
                                                            note.addBendPoint(new window.alphaTab.model.BendPoint(0, 0));
                                                            note.addBendPoint(new window.alphaTab.model.BendPoint(6, 4));
                                                            note.addBendPoint(new window.alphaTab.model.BendPoint(12, 4));
                                                        }
                                                        applied.bend++;
                                                    } catch { /* */ }
                                                } else if (tech === "vibrato" && VT) {
                                                    try {
                                                        beat.vibrato = VT.Slight;
                                                        applied.vibrato++;
                                                    } catch { /* */ }
                                                } else if (tech === "let_ring") {
                                                    try {
                                                        note.isLetRing = true;
                                                        applied.let_ring++;
                                                    } catch { /* */ }
                                                } else if (tech === "ghost_note") {
                                                    try {
                                                        note.isGhost = true;
                                                        applied.ghost_note++;
                                                    } catch { /* */ }
                                                } else if (tech === "trill") {
                                                    try {
                                                        note.trillFret = note.fret + 2; // デフォルトで全音上のトリル
                                                        if (window.alphaTab.model?.Duration) {
                                                            note.trillSpeed = window.alphaTab.model.Duration.ThirtySecond;
                                                        }
                                                        applied.trill++;
                                                    } catch { /* */ }
                                                }
                                                noteIdx++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        console.log(`[TabView] Techniques applied: ${JSON.stringify(applied)} / ${techMap.length} notes`);
                    } catch (e) {
                        console.warn("[TabView] Technique apply error:", e);
                    }
                };

                api.renderStarted.on(() => setLoading(true));
                api.renderFinished.on(() => {
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
                    // Scroll to top
                    if (containerRef.current) containerRef.current.scrollTop = 0;
                });
                api.error.on((e) => {
                    console.error("[AlphaTab Error]", e);
                    if (!destroyed) { setError("TAB表示エラー"); setLoading(false); }
                });

                // scoreLoadedイベント: テクニック適用後に再レンダリング
                let techApplied = false;
                api.scoreLoaded.on(async (score) => {
                    if (techApplied || destroyed) return;
                    techApplied = true;
                    await applyTechniques(score);
                    // テクニック適用後に再レンダリング
                    api.render();
                });

                const encoder = new TextEncoder();
                const data = encoder.encode(xmlText);
                api.load(data instanceof Uint8Array ? data : new Uint8Array(data));
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
    }, [sessionId, apiBase, transpose, capo]);

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

            if (nowPlaying && !wasPlaying && container && ms < 1000) {
                container.scrollTo({ top: 0, behavior: "instant" });
            }
            wasPlaying = nowPlaying;

            if (cursor && boundsReadyRef.current) {
                const beat = findBeat(ms);
                if (beat) {
                    const { x, y, h } = beat.vb;
                    cursor.style.display = "block";
                    cursor.style.left = `${x}px`;
                    cursor.style.top = `${y}px`;
                    cursor.style.width = `${beat.vb.w}px`;
                    cursor.style.height = `${h}px`;

                    // Auto-scroll: autoScrollRef.current がtrueの時のみ追従
                    if (nowPlaying && container && autoScrollRef.current) {
                        const now = Date.now();
                        if (now - lastScrollMs > 400) {
                            const cursorScreenY = y - container.scrollTop;
                            const viewH = container.clientHeight;
                            if (cursorScreenY < 0 || cursorScreenY > viewH * 0.55) {
                                const targetTop = Math.max(0, y - viewH * 0.3);
                                container.scrollTo({ top: targetTop, behavior: "smooth" });
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
        <div
            ref={containerRef}
            className="tab-print-container"
            style={{
                width: "100%", height: "100%",
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
                        background: "rgba(59,130,246,0.5)",
                        borderRadius: 1,
                        transition: "left 0.08s linear, top 0.15s ease-out, width 0.08s linear, height 0.1s ease",
                        willChange: "left, top, width, height",
                    }}
                />
                {/* AlphaTab renders into this div */}
                <div ref={wrapperRef} className="alpha-tab-wrapper" style={{ width: "100%", minHeight: "100vh" }} />

                {/* ノート編集ポップアップ */}
                {editNote && (
                    <div
                        style={{
                            position: "absolute",
                            left: Math.max(10, editNote.x - 60), top: editNote.y + 20,
                            zIndex: 100,
                            background: "rgba(20,20,30,0.95)",
                            border: "2px solid #3b82f6",
                            borderRadius: 12,
                            padding: "10px 14px",
                            boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
                            display: "flex", flexDirection: "column", gap: 6,
                            minWidth: 120,
                        }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600 }}>
                            🎸 弦{editNote.string} フレット{editNote.fret}
                        </div>
                        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                            <input
                                ref={editInputRef}
                                type="number"
                                min="0" max="24"
                                value={editInput}
                                onChange={(e) => setEditInput(e.target.value)}
                                onKeyDown={async (e) => {
                                    if (e.key === "Enter") {
                                        e.preventDefault();
                                        const newFret = parseInt(editInput);
                                        if (isNaN(newFret) || newFret < 0 || newFret > 24) return;
                                        setEditSaving(true);
                                        try {
                                            await fetch(`${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`, {
                                                method: "PATCH",
                                                headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({ noteIndex: editNote.noteIndex, fret: newFret }),
                                            });
                                            setEditNote(null);
                                            initKeyRef.current = null; // 再読み込み強制
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
                                    if (isNaN(newFret) || newFret < 0 || newFret > 24) return;
                                    setEditSaving(true);
                                    try {
                                        await fetch(`${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`, {
                                            method: "PATCH",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ noteIndex: editNote.noteIndex, fret: newFret }),
                                        });
                                        setEditNote(null);
                                        initKeyRef.current = null;
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
                                            body: JSON.stringify({ noteIndex: editNote.noteIndex, delete: true }),
                                        });
                                        setEditNote(null);
                                        initKeyRef.current = null;
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
                        <div style={{ fontSize: 10, color: "#64748b" }}>Enter=保存 Esc=閉じる</div>
                    </div>
                )}
            </div>

            {/* Auto-scroll toggle */}
            <div
                className="auto-scroll-btn"
                style={{
                    position: "fixed", bottom: 24, right: 24, zIndex: 50,
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

            <style>{`
                .at-cursor-beat, .at-cursor-bar, .at-selection, .at-highlight { display: none !important; }
                .alphaTabSurface { position: static !important; }
                /* TAB表示改善 */
                .at-surface { overflow: visible !important; }
                .at-main text { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
                .at-staff-tab .at-note-number { font-size: 13px !important; font-weight: 600 !important; }
                /* テクニック記号の読みやすさ */
                .at-effect-note text { font-size: 10px !important; fill: #555 !important; }
                /* 行間を確保 */
                .at-system { margin-bottom: 8px !important; }
                /* 印刷用 */
                @media print {
                    body { background: white !important; }
                    .at-surface { transform: scale(1) !important; width: 100% !important; overflow: visible !important; }
                    .at-system { break-inside: avoid !important; page-break-inside: avoid !important; margin-bottom: 16px !important; display: block !important; }
                    .at-viewport { overflow: visible !important; height: auto !important; display: block !important; }
                    .tab-print-container { height: auto !important; overflow: visible !important; padding-bottom: 0 !important; }
                }
            `}</style>
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

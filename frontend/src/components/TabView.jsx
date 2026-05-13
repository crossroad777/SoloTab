import React, { useEffect, useRef, useState } from "react";
// ScoreToolbar: /score API未実装のため一時無効化 (コンポーネントはファイルとして残存)
// import ScoreToolbar from "./ScoreToolbar";

/**
 * TabView — AlphaTab TAB 譜表示 + GP5ダウンロード
 * - AlphaTab組み込みカーソルをtickPositionで駆動（音声同期）
 * - beats.json → tick変換テーブルで非線形テンポにも対応
 * - ノートクリック編集UI
 */
const TabViewInner = ({ sessionId, apiBase, currentTime, isPlaying, transpose = 0, capo = 0, onApiReady }) => {
    const containerRef = useRef(null);
    const wrapperRef = useRef(null);
    const apiRef = useRef(null);
    const timeRef = useRef(0);
    const playingRef = useRef(false);
    const initKeyRef = useRef(null);
    const tickMapRef = useRef(null); // {beats:[], bpm:number, ticksPerBeat:960}

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
    // Time→Tick変換: beats.jsonの実時刻をAlphaTabのtick位置に変換
    // ============================================================
    const TICKS_PER_BEAT = 960; // AlphaTab標準

    const timeToTick = (audioSec) => {
        const tm = tickMapRef.current;
        if (!tm || !tm.beats || tm.beats.length === 0) {
            // フォールバック: 線形変換
            return Math.max(0, audioSec * (tm?.bpm || 120) / 60 * TICKS_PER_BEAT);
        }
        const beats = tm.beats;
        // audioSecがどの拍間にあるか二分探索
        if (audioSec <= beats[0]) return 0;
        if (audioSec >= beats[beats.length - 1]) {
            return (beats.length - 1) * TICKS_PER_BEAT;
        }
        let lo = 0, hi = beats.length - 2;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (audioSec >= beats[mid] && audioSec < beats[mid + 1]) {
                // 拍内を線形補間
                const frac = (audioSec - beats[mid]) / (beats[mid + 1] - beats[mid]);
                return (mid + frac) * TICKS_PER_BEAT;
            }
            if (audioSec < beats[mid]) hi = mid - 1;
            else lo = mid + 1;
        }
        return lo * TICKS_PER_BEAT;
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
                        tickMapRef.current = {
                            beats: beatData.beats || [],
                            bpm: beatData.bpm || 120,
                        };
                        console.log(`[TabView] TickMap: ${tickMapRef.current.beats.length} beats, ${tickMapRef.current.bpm} BPM`);
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

                if (settings.display.resources) {
                    settings.display.resources.titleFont = new window.alphaTab.model.Font("Arial", 16, 1);
                }

                // === Player + 組み込みカーソル有効化 ===
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
                    const px = (evt?.pageX || evt?.clientX || 200) - rect.left;
                    const py = (evt?.pageY || evt?.clientY || 200) - rect.top + containerRef.current.scrollTop;

                    setEditNote({ noteIndex: noteIdx, fret: note.fret, string: note.string, x: px, y: py, alphaNote: note });
                    setEditInput(String(note.fret));
                    setTimeout(() => editInputRef.current?.focus(), 50);
                });

                api.renderStarted.on(() => setLoading(true));
                api.renderFinished.on(() => {
                    if (destroyed) return;
                    setLoading(false);

                    // タイトル上書き（renderFinished時にapi.scoreが確実に存在）
                    if (songTitle && api.score) {
                        api.score.title = songTitle;
                        api.score.artist = 'SoloTab';
                    }

                    if (containerRef.current) containerRef.current.scrollTop = 0;
                    console.log(`[TabView] Render finished, cursor ready`);
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
    // Sync loop: AlphaTab組み込みカーソルをtickPositionで駆動
    // ============================================================
    useEffect(() => {
        let animId;
        let lastTick = -1;

        const sync = () => {
            const api = apiRef.current;
            if (api && tickMapRef.current) {
                const sec = timeRef.current;
                const tick = timeToTick(sec);
                // 変化が小さいときはスキップ（負荷軽減）
                if (Math.abs(tick - lastTick) > 10) {
                    lastTick = tick;
                    try { api.tickPosition = tick; } catch { /* ignore */ }
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

            {/* Score container */}
            <div style={{ position: "relative", padding: 0, margin: 0 }}>
                {/* AlphaTab renders into this div (組み込みカーソル使用) */}
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

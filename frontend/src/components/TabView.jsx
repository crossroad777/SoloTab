import React, { useEffect, useRef, useState, useCallback } from "react";
import ScoreToolbar from "./ScoreToolbar";

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
    const surfaceOffsetRef = useRef({ x: 0, y: 0 });

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [autoScroll, setAutoScroll] = useState(true);
    const autoScrollRef = useRef(true);
    const [scale, setScale] = useState(0.75);
    const scaleRef = useRef(0.75);

    // --- Score Player state ---
    const [scorePlayerReady, setScorePlayerReady] = useState(false);
    const [scorePlayerState, setScorePlayerState] = useState(0); // 0=stopped, 1=playing, 2=paused

    // --- TAB編集UI state ---
    const [editNote, setEditNote] = useState(null); // {noteIndex, fret, string, x, y}
    const [editInput, setEditInput] = useState("");
    const [editSaving, setEditSaving] = useState(false);
    const [reloadKey, setReloadKey] = useState(0);
    const editInputRef = useRef(null);

    // --- Score Model state ---
    const [scoreData, setScoreData] = useState(null);

    const fetchScore = useCallback(async () => {
        if (!sessionId) return;
        try {
            const res = await fetch(`${apiBase}/result/${sessionId}/score`);
            if (res.ok) setScoreData(await res.json());
        } catch (e) { console.warn("Score fetch failed:", e); }
    }, [sessionId, apiBase]);

    useEffect(() => { fetchScore(); }, [fetchScore]);

    const handleScoreUpdate = useCallback(() => {
        fetchScore();
        // MusicXMLが再生成されたのでAlphaTabをリロード
        setReloadKey(k => k + 1);
        initKeyRef.current = null;
    }, [fetchScore]);

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
        // Y/高さはスタッフシステム（1段）の境界を使用し、X/幅は個別小節から取得
        // → エフェクト記号(let ring, vibrato等)による小節ごとのY変動を排除
        const barCoords = [];
        for (const system of systems) {
            const sgBars = system.bars;
            if (!sgBars) continue;
            const sysVb = system.visualBounds;
            for (const sgBar of sgBars) {
                const barIndex = sgBar.index;
                const barVb = sgBar.visualBounds;
                if (barIndex == null || !barVb || barVb.x == null) continue;
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
                // Cache-bust + retry (Volume commit遅延対策)
                let res;
                for (let attempt = 0; attempt < 3; attempt++) {
                    res = await fetch(`${apiBase}/result/${sessionId}/musicxml?t=${Date.now()}`);
                    if (res.ok) break;
                    console.warn(`[TabView] musicxml attempt ${attempt + 1} failed, retrying...`);
                    await new Promise(r => setTimeout(r, 1500));
                }
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
                settings.core.fontDirectory = "https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/font/";

                // === 五線譜 + TAB 2段表示 ===
                // ScoreTab: 五線譜+TABの両方を強制描画する
                // フレット値はscoreLoadedイベント後にMusicXMLの値で上書き
                settings.display.staveProfile = window.alphaTab.StaveProfile.ScoreTab;
                settings.display.layoutMode = window.alphaTab.LayoutMode.Page;
                settings.display.scale = scaleRef.current;   // ズームレベル（state管理）
                settings.display.stretchForce = 1.2;        // ノート間スペースを広げる
                settings.display.barsPerRow = 4;             // 1行4小節で読みやすく

                // === 記譜設定 ===
                settings.notation.rhythmMode = 0;            // TABリズム表記OFF（視認性向上）
                settings.notation.fingeringMode = 0;         // 運指表記OFF

                // テクニック記号の表示はAlphaTabのデフォルト設定に依存
                // MusicXMLのharmonic/staccato/bend/slide要素は自動検出される

                // タイトルフォントを小さくして長い曲名も収まるように
                if (settings.display.resources) {
                    settings.display.resources.titleFont = new window.alphaTab.model.Font("Arial", 16, 1); // 1=Bold
                }

                // === Player: boundsLookup生成に必要なので有効化、カーソルは無効 ===
                settings.player.enablePlayer = true;
                settings.player.enableCursor = false; // AlphaTabカーソルOFF（カスタムカーソルを使用）
                settings.player.scrollMode = 0; // スクロールOFF（横シフト防止）
                settings.player.soundFont = "https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.0/dist/soundfont/sonivox.sf2";
                settings.core.includeNoteBounds = true; // boundsLookup + ノートクリック検出に必要

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

                    // AlphaTab surface要素のオフセットを計測（カーソル補正用）
                    try {
                        const wrapper = wrapperRef.current;
                        if (wrapper) {
                            // AlphaTabが生成するcanvas/svg surface要素を探す
                            const surface = wrapper.querySelector('.at-surface, canvas, svg');
                            if (surface) {
                                const wrapperRect = wrapper.getBoundingClientRect();
                                const surfaceRect = surface.getBoundingClientRect();
                                surfaceOffsetRef.current = {
                                    x: surfaceRect.left - wrapperRect.left,
                                    y: surfaceRect.top - wrapperRect.top,
                                };
                                console.log(`[TabView] Surface offset: x=${surfaceOffsetRef.current.x}, y=${surfaceOffsetRef.current.y}`);
                            }
                        }
                    } catch { /* noop */ }

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

                // --- フレット/弦オーバーライド ---
                // AlphaTabはpitchからフレットを自動計算するが、
                // MusicXMLの<technical>値と一致しない。scoreLoaded後に
                // MusicXMLのfret/string値で上書きする。
                const parser = new DOMParser();
                const xmlDoc = parser.parseFromString(xmlText, "text/xml");
                const xmlNotes = xmlDoc.querySelectorAll("note");
                const tabFrets = [];
                for (const noteEl of xmlNotes) {
                    const staffEl = noteEl.querySelector("staff");
                    if (!staffEl || staffEl.textContent !== "2") continue;
                    if (noteEl.querySelector("rest")) continue;
                    const fretEl = noteEl.querySelector("fret");
                    const stringEl = noteEl.querySelector("string");
                    if (fretEl && stringEl) {
                        tabFrets.push({
                            fret: parseInt(fretEl.textContent),
                            string: parseInt(stringEl.textContent),
                        });
                    }
                }
                console.log(`[TabView] Parsed ${tabFrets.length} TAB fret/string values from MusicXML`);

                let fretOverrideApplied = false;
                api.scoreLoaded.on((score) => {
                    if (fretOverrideApplied || destroyed) return;
                    fretOverrideApplied = true;
                    try {
                        const track = score.tracks[0];
                        if (!track) return;
                        // 全stavesのfret/stringを上書き（五線譜+TAB両方）
                        let totalOverridden = 0;
                        for (const staff of track.staves) {
                            let noteIdx = 0;
                            for (const bar of staff.bars) {
                                for (const voice of bar.voices) {
                                    for (const beat of voice.beats) {
                                        if (beat.isRest) continue;
                                        for (const note of beat.notes) {
                                            if (noteIdx < tabFrets.length) {
                                                const target = tabFrets[noteIdx];
                                                if (note.fret !== target.fret || note.string !== target.string) {
                                                    note.fret = target.fret;
                                                    note.string = target.string;
                                                    totalOverridden++;
                                                }
                                            }
                                            noteIdx++;
                                        }
                                    }
                                }
                            }
                            console.log(`[TabView] Staff fret override: ${noteIdx} notes processed`);
                        }
                        console.log(`[TabView] Total fret override: ${totalOverridden} notes corrected`);
                        if (totalOverridden > 0) {
                            api.render();
                        }
                    } catch (e) {
                        console.warn("[TabView] Fret override error:", e);
                    }
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

            if (nowPlaying && !wasPlaying && container && ms < 1000) {
                container.scrollTo({ top: 0, behavior: "instant" });
            }
            wasPlaying = nowPlaying;

            if (cursor && boundsReadyRef.current) {
                const beat = findBeat(ms);
                if (beat) {
                    const { x, y, h } = beat.vb;
                    const off = surfaceOffsetRef.current;
                    cursor.style.display = "block";
                    cursor.style.left = `${x + off.x}px`;
                    cursor.style.top = `${y + off.y}px`;
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
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
            {/* PowerTab互換ツールバー */}
            <ScoreToolbar
                sessionId={sessionId}
                apiBase={apiBase}
                score={scoreData}
                onScoreUpdate={handleScoreUpdate}
            />
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

            {/* スコアプレーヤー + ズームコントロール + Auto-scroll */}
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

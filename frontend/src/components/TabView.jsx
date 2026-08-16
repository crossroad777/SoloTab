import React, { useEffect, useRef, useState } from "react";
import ScoreToolbar from "./ScoreToolbar";
import { exportToPDF } from "../utils/pdfExport";

/**
 * TabView — AlphaTab TAB 譜表示
 * - カスタムBeatMapでtick→座標マッピング
 * - カスタム青カーソルバー + オートスクロール
 */
const TabViewInner = ({ sessionId, apiBase, currentTime, isPlaying, transpose = 0, capo = 0, metronomeEnabled = false, syncOffset = 0, tempoMultiplier = 1.0, onApiReady, onNoteEdited }) => {
    const containerRef = useRef(null);
    const wrapperRef = useRef(null);
    const cursorRef = useRef(null);
    const apiRef = useRef(null);
    const beatMapRef = useRef([]);
    const boundsReadyRef = useRef(false);
    const timeRef = useRef(0);
    const playingRef = useRef(false);
    const initKeyRef = useRef(null);
    const beatsDataRef = useRef([]);
    const chordsDataRef = useRef([]);

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [autoScroll, setAutoScroll] = useState(true);
    const autoScrollRef = useRef(true);
    const [scale, setScale] = useState(0.75);
    const scaleRef = useRef(0.75);



    // --- TAB編集UI state ---
    const [editNote, setEditNote] = useState(null); // {noteIndex, fret, string, x, y}
    const [editSaving, setEditSaving] = useState(false);
    const [editInput, setEditInput] = useState("");
    const editInputRef = useRef(null);
    const [reloadKey, setReloadKey] = useState(0);
    const [canUndo, setCanUndo] = useState(false);
    const [canRedo, setCanRedo] = useState(false);
    
    const [isExporting, setIsExporting] = useState(false);

    const handlePdfExport = async () => {
        if (!containerRef.current || isExporting) return;
        setIsExporting(true);
        try {
            const api = apiRef.current;
            const title = api?.score?.title || "SoloTab_Score";
            const artist = api?.score?.artist || "";
            const filename = artist ? `${artist} - ${title}.pdf` : `${title}.pdf`;
            
            await exportToPDF(containerRef.current, filename);
        } catch (e) {
            console.error("PDF Export error:", e);
            alert("PDFエクスポートに失敗しました");
        } finally {
            setIsExporting(false);
        }
    };

    const handleUndo = async () => {
        if (!canUndo) return;
        setLoading(true);
        try {
            const res = await fetch(`${apiBase}/result/${sessionId}/undo`, { method: "POST" });
            if (res.ok) {
                await new Promise(r => setTimeout(r, 500));
                onNoteEdited?.();
                setReloadKey(k => k + 1);
            }
        } catch (e) { console.error(e); }
        setLoading(false);
    };

    const handleRedo = async () => {
        if (!canRedo) return;
        setLoading(true);
        try {
            const res = await fetch(`${apiBase}/result/${sessionId}/redo`, { method: "POST" });
            if (res.ok) {
                await new Promise(r => setTimeout(r, 500));
                onNoteEdited?.();
                setReloadKey(k => k + 1);
            }
        } catch (e) { console.error(e); }
        setLoading(false);
    };

    useEffect(() => {
        const handleKeyDown = async (e) => {
            // INPUT/TEXTAREA要素での発火を防ぐ
            const tag = e.target?.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            
            const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
            const ctrlOrMeta = isMac ? e.metaKey : e.ctrlKey;
            
            if (ctrlOrMeta && e.key === 'z' && !e.shiftKey) {
                e.preventDefault();
                handleUndo();
            } else if (ctrlOrMeta && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
                e.preventDefault();
                handleRedo();
            } else if (editNote) {
                if (e.key === 'Escape') {
                    e.preventDefault();
                    setEditNote(null);
                } else if (e.key === 'Delete' || e.key === 'Backspace') {
                    e.preventDefault();
                    if (editNote.noteIndex !== -1) {
                        if (!confirm("このノートを削除しますか？")) return;
                        setEditSaving(true);
                        try {
                            const res = await fetch(`${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`, {
                                method: "PATCH",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ delete: true, start_time: editNote.startTime, string: editNote.string, old_fret: editNote.fret }),
                            });
                            if (res.ok) {
                                setEditNote(null);
                                await new Promise(r => setTimeout(r, 300));
                                onNoteEdited?.();
                            }
                        } catch (err) { console.error("Save failed:", err); }
                        setEditSaving(false);
                    }
                } else if (e.key === 'l' || e.key === 'L') {
                    e.preventDefault();
                    if (editNote.noteIndex !== -1) {
                        setEditSaving(true);
                        try {
                            const newAnchor = !editNote.editAnchor;
                            const res = await fetch(`${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`, {
                                method: "PATCH",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    anchor: newAnchor,
                                    start_time: editNote.startTime,
                                    string: editNote.string,
                                    old_fret: editNote.fret,
                                    fret: editNote.fret,
                                    finger: editNote.editFinger
                                }),
                            });
                            if (res.ok) {
                                setEditNote(null);
                                await new Promise(r => setTimeout(r, 300));
                                onNoteEdited?.();
                            }
                        } catch (err) { console.error("Save failed:", err); }
                        setEditSaving(false);
                    }
                }
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [canUndo, canRedo, editNote, apiBase, sessionId, onNoteEdited]);


    useEffect(() => {
        timeRef.current = currentTime;
        playingRef.current = isPlaying;
    }, [currentTime, isPlaying]);

    const syncOffsetRef = useRef(syncOffset);
    const tempoMultiplierRef = useRef(tempoMultiplier);
    const metronomeEnabledRef = useRef(metronomeEnabled);

    useEffect(() => {
        syncOffsetRef.current = syncOffset;
        tempoMultiplierRef.current = tempoMultiplier;
        metronomeEnabledRef.current = metronomeEnabled;
    }, [syncOffset, tempoMultiplier, metronomeEnabled]);

    // --- Web Audio API Metronome ---
    const audioCtxRef = useRef(null);
    const nextBeatIdxRef = useRef(0);
    const scheduledOscsRef = useRef([]);

    useEffect(() => {
        if (isPlaying && metronomeEnabled && !audioCtxRef.current) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (AudioContext) {
                audioCtxRef.current = new AudioContext();
            }
        }
        if (!isPlaying) {
            // When paused, we will recalculate nextBeatIdx upon resume
            nextBeatIdxRef.current = -1;
            scheduledOscsRef.current.forEach(osc => { try { osc.stop(); } catch(e){} });
            scheduledOscsRef.current = [];
        }
    }, [isPlaying, metronomeEnabled]);

    // ============================================================
    // BeatMap: 小節単位 — beats.jsonの実時刻 + AlphaTab bar座標
    // ============================================================
    // notesDataRef: APIから取得したノートデータ（start時刻あり）
    const notesDataRef = useRef([]);
    const getTabY = (api, vbY, vbH) => {
        const bl = api?.renderer?.boundsLookup;
        const masterBars = bl?.masterBars ?? [];
        const tabYMap = new Map();
        for (const mb of masterBars) {
            const sysY = mb.visualBounds?.y;
            if (sysY == null || tabYMap.has(sysY)) continue;
            const tabStaveVb = mb.bars?.[0]?.bars?.[1]?.visualBounds
                            ?? mb.bars?.[0]?.bars?.[0]?.visualBounds;
            if (tabStaveVb?.y != null) {
                tabYMap.set(sysY, tabStaveVb.y - 8);
            } else {
                tabYMap.set(sysY, sysY + (mb.visualBounds?.h ?? 0) * 0.77);
            }
        }
        if (tabYMap.has(vbY)) return tabYMap.get(vbY);
        let best = null, bestDist = Infinity;
        for (const [sy] of tabYMap) {
            const d = Math.abs(sy - vbY);
            if (d < bestDist) { bestDist = d; best = sy; }
        }
        return best != null ? tabYMap.get(best) : (vbY + vbH * 0.77);
    };

    const handleWrapperClick = (e, api) => {
        if (editSaving) return;

        const wrapper = wrapperRef.current;
        if (!wrapper || !containerRef.current) return;

        const rect = wrapper.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;

        // 1. まず、クリックされた位置に既存の音符があるか座標ベースで判定する
        const lookup = api?.renderer?.boundsLookup;
        let matchedNoteInfo = null;

        if (lookup) {
            const groups = lookup.staffSystems || lookup.staveGroups || [];
            const margin = 10; // クリック判定の許容マージン (px)
            
            outer: for (const sys of groups) {
                const bars = sys.bars || sys.masterBars || [];
                for (const bar of bars) {
                    const barBounds = bar.bars || [];
                    for (const bb of barBounds) {
                        const beats = bb.beats || [];
                        for (const beatBounds of beats) {
                            const notes = beatBounds.notes || [];
                            for (const nb of notes) {
                                if (nb.note) {
                                    const bounds = nb.noteHeadBounds || beatBounds.visualBounds;
                                    if (bounds) {
                                        const { x, y, w, h } = bounds;
                                        if (px >= x - margin && px <= x + w + margin &&
                                            py >= y - margin && py <= y + h + margin) {
                                            matchedNoteInfo = {
                                                note: nb.note,
                                                bounds: bounds
                                            };
                                            break outer;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. 既存の音符が見つかった場合 -> 編集ポップアップを開く
        if (matchedNoteInfo) {
            const { note, bounds } = matchedNoteInfo;
            const score = api.score;
            if (!score) return;

            const beat = note.beat;
            const ticksPerBeat = 960;
            const bpm = (typeof score.tempo === 'object' && score.tempo !== null)
                ? (score.tempo.value || 120)
                : (score.tempo || 120);
            const approxTimeSec = (beat.absolutePlaybackStart / ticksPerBeat) * (60 / bpm);
            const backendString = 7 - note.string;

            // バックエンドのノートインデックスを特定
            let backendIdx = -1;
            const backendNotes = notesDataRef.current;
            if (backendNotes.length > 0) {
                let bestDist = Infinity;
                for (let i = 0; i < backendNotes.length; i++) {
                    const bn = backendNotes[i];
                    const startVal = bn.start_time ?? bn.start ?? 0;
                    if (Number(bn.string) === backendString && Number(bn.fret) === Number(note.fret)) {
                        const d = Math.abs(startVal - approxTimeSec);
                        if (d < bestDist) { bestDist = d; backendIdx = i; }
                    }
                }
                if (backendIdx < 0) {
                    let bestDist2 = Infinity;
                    for (let i = 0; i < backendNotes.length; i++) {
                        const bn = backendNotes[i];
                        const startVal = bn.start_time ?? bn.start ?? 0;
                        const d = Math.abs(startVal - approxTimeSec);
                        if (d < bestDist2 && Number(backendNotes[i].string) === backendString) {
                            bestDist2 = d; backendIdx = i;
                        }
                    }
                }
            }

            if (backendIdx >= 0) {
                const matchedNote = backendNotes[backendIdx];
                const containerRect = containerRef.current.getBoundingClientRect();
                const popupX = e.clientX - containerRect.left;
                const popupY = e.clientY - containerRect.top + containerRef.current.scrollTop;

                console.log(`[TabView] Note clicked via coordinates: fret=${note.fret} str=${backendString} idx=${backendIdx} t≈${approxTimeSec.toFixed(2)}s`);
                setEditNote({
                    noteIndex: backendIdx,
                    fret: note.fret,
                    string: backendString,
                    editFinger: matchedNote?.left_hand_finger || 0,
                    editAnchor: matchedNote?._is_anchor || false,
                    startTime: matchedNote?.start,
                    x: popupX,
                    y: popupY,
                    alphaNote: note
                });
                setEditInput(String(note.fret));
                setTimeout(() => editInputRef.current?.focus(), 50);
            } else {
                console.warn('[TabView] Could not find matching backend note for coordinates');
            }
            return;
        }

        // 3. 既存の音符がない場合 -> 新規ノート追加ポップアップを開く
        const map = beatMapRef.current;
        if (!map || !map.length) return;

        let bestBeat = null;
        let bestDist = Infinity;

        for (const entry of map) {
            const { x, y, w, h } = entry.vb;
            const inY = py >= y && py <= y + h;
            if (inY) {
                const centerX = x + w / 2;
                const dist = Math.abs(px - centerX);
                if (dist < bestDist && dist < w * 30) {
                    bestDist = dist;
                    bestBeat = entry;
                }
            }
        }

        if (!bestBeat) {
            for (const entry of map) {
                const { x, y, w, h } = entry.vb;
                const centerX = x + w / 2;
                const centerY = y + h / 2;
                const dist = Math.sqrt(Math.pow(px - centerX, 2) + Math.pow(py - centerY, 2));
                if (dist < bestDist && dist < 300) {
                    bestDist = dist;
                    bestBeat = entry;
                }
            }
        }

        if (bestBeat) {
            const tabTopY = getTabY(api, bestBeat.vb.y, bestBeat.vb.h);
            const offset = py - tabTopY;
            let estString = 1;
            if (offset < 13) estString = 1;
            else if (offset < 23) estString = 2;
            else if (offset < 33) estString = 3;
            else if (offset < 43) estString = 4;
            else if (offset < 53) estString = 5;
            else estString = 6;

            const startTimeSec = bestBeat.startMs / 1000;
            const containerRect = containerRef.current.getBoundingClientRect();
            const popupX = e.clientX - containerRect.left;
            const popupY = e.clientY - containerRect.top + containerRef.current.scrollTop;

            setEditNote({
                noteIndex: -1, // 新規追加を示すマーク
                fret: 0,
                string: estString,
                startTime: startTimeSec,
                x: popupX,
                y: popupY
            });
            setEditInput("0");
            setTimeout(() => editInputRef.current?.focus(), 50);
        }
    };


    const buildBeatMap = (api) => {
        if (!api.score || !api.renderer?.boundsLookup) {
            console.log(`[TabView] BeatMap: score=${!!api.score}, boundsLookup=${!!api.renderer?.boundsLookup}`);
            return false;
        }
        const lookup = api.renderer.boundsLookup;
        const systems = lookup.staffSystems || lookup.staveGroups;
        if (!systems || systems.length === 0) {
            console.warn(`[TabView] BeatMap: no staffSystems/staveGroups`);
            return false;
        }

        const notes = notesDataRef.current;
        // AlphaTabのtempoプロパティを安全に数値として取得
        let rawTempo = api.score.tempo;
        let bpm = 120;
        if (typeof rawTempo === 'number' && rawTempo > 0) {
            bpm = rawTempo;
        } else if (typeof rawTempo === 'object' && rawTempo !== null && typeof rawTempo.value === 'number') {
            bpm = rawTempo.value;
        } else if (typeof api.score.tempoValue === 'number') {
            bpm = api.score.tempoValue;
        }

        const ticksPerBeat = 960;
        const beatsArr = beatsDataRef.current; // beats.json実時刻(秒)

        // ===================================================================
        // Strategy 1: tick-based mapping using AlphaTab beat model
        // Each boundsLookup beat has .beat → absolutePlaybackStart (ticks)
        // Convert ticks directly to beatIdx: beatIdx = ticks / 960
        // ===================================================================
        const tickEntries = [];
        for (const system of systems) {
            const sgBars = system.bars || system.masterBars;
            if (!sgBars) continue;
            const sysVb = system.visualBounds;
            for (const sgBar of sgBars) {
                const innerBars = sgBar.bars || [];
                for (const barBounds of innerBars) {
                    for (const beatBounds of (barBounds.beats || [])) {
                        const bvb = beatBounds.visualBounds || beatBounds.bounds;
                        if (!bvb || bvb.x == null) continue;
                        const beat = beatBounds.beat;
                        const ticks = beat?.absolutePlaybackStart ?? beat?.playbackStart ?? null;
                        if (ticks != null && !isNaN(ticks)) {
                            tickEntries.push({
                                ticks: ticks,
                                timeMs: (ticks / ticksPerBeat) * (60000 / bpm),
                                isRest: beat.isRest || beat.isEmpty || false,
                                vb: {
                                    x: bvb.x,
                                    y: sysVb ? sysVb.y : bvb.y,
                                    w: bvb.w,
                                    h: sysVb ? sysVb.h : bvb.h,
                                },
                            });
                        }
                    }
                }
            }
        }

        if (tickEntries.length > 0) {
            tickEntries.sort((a, b) => a.ticks - b.ticks);

            // =====================================================
            // Piecewise calibration using beats.json
            // Each GP5 beat (fixed BPM tick) maps to a real audio
            // time from beats.json, handling rubato/tempo variation.
            // =====================================================
            const tickToRealMs = (ticks) => {
                if (beatsArr.length < 2) {
                    // Fallback: linear from tempo
                    return (ticks / ticksPerBeat) * (60000 / bpm);
                }
                // ticks → which beat index in GP5 (960 ticks = 1 beat)
                const beatIdx = ticks / ticksPerBeat;
                const lo = Math.floor(beatIdx);
                const frac = beatIdx - lo;

                // Clamp to available beats
                if (lo >= beatsArr.length - 1) {
                    const lastIdx = beatsArr.length - 1;
                    const lastInterval = (beatsArr[lastIdx] - beatsArr[Math.max(0, lastIdx - 1)]) * 1000;
                    return beatsArr[lastIdx] * 1000 + (beatIdx - lastIdx) * lastInterval;
                }
                if (lo < 0) return beatsArr[0] * 1000;

                // Interpolate between beat[lo] and beat[lo+1]
                const tLo = beatsArr[lo] * 1000;
                const tHi = beatsArr[Math.min(lo + 1, beatsArr.length - 1)] * 1000;
                return tLo + frac * (tHi - tLo);
            };

            // Apply note-based offset if beats.json not available
            let useBeats = beatsArr.length >= 2;
            let offsetMs = 0;
            let linearScale = 1;
            if (!useBeats && notes.length > 0) {
                const sorted = [...notes].sort((a, b) => (a.start ?? a.start_time ?? 0) - (b.start ?? b.start_time ?? 0));
                const firstStart = sorted[0].start ?? sorted[0].start_time ?? 0;
                const lastStart = sorted[sorted.length - 1].start ?? sorted[sorted.length - 1].start_time ?? 0;
                const firstNoteMs = firstStart * 1000;
                const lastNoteMs = lastStart * 1000;
                const nonRests = tickEntries.filter(e => !e.isRest);
                if (nonRests.length >= 2) {
                    const firstTickMs = nonRests[0].timeMs;
                    const lastTickMs = nonRests[nonRests.length - 1].timeMs;
                    const tickSpan = lastTickMs - firstTickMs;
                    const noteSpan = lastNoteMs - firstNoteMs;
                    if (tickSpan > 0 && noteSpan > 0) {
                        linearScale = noteSpan / tickSpan;
                        offsetMs = firstNoteMs - firstTickMs * linearScale;
                    }
                }
            }

            const map = [];
            let lastStartMs = -1;
            for (let i = 0; i < tickEntries.length; i++) {
                let startMs = useBeats
                    ? tickToRealMs(tickEntries[i].ticks)
                    : tickEntries[i].timeMs * linearScale + offsetMs;

                // 単調増加性の厳密保証（逆戻り・ワープを完全防止）
                if (startMs <= lastStartMs) {
                    startMs = lastStartMs + 1;
                }
                lastStartMs = startMs;

                const nextRawMs = (i + 1 < tickEntries.length)
                    ? (useBeats
                        ? tickToRealMs(tickEntries[i + 1].ticks)
                        : tickEntries[i + 1].timeMs * linearScale + offsetMs)
                    : startMs + 600;

                const endMs = Math.max(startMs + 50, nextRawMs);
                map.push({ startMs, endMs, vb: tickEntries[i].vb, ticks: tickEntries[i].ticks });
            }

            beatMapRef.current = map;
            console.log(`[TabView] BeatMap (${useBeats ? 'beats.json piecewise' : 'linear'}): ${map.length} entries (strictly monotonic), ` +
                `range: ${(map[0].startMs/1000).toFixed(2)}s → ${(map[map.length-1].startMs/1000).toFixed(2)}s`);
            return true;
        }

        // ===================================================================
        // Strategy 2 fallback: note-time to visual-coord mapping
        // If tick data unavailable, match note events to beat coords by index
        // ===================================================================
        const beatCoords = [];
        for (const system of systems) {
            const sgBars = system.bars || system.masterBars;
            if (!sgBars) continue;
            const sysVb = system.visualBounds;
            for (const sgBar of sgBars) {
                const innerBars = sgBar.bars || [];
                for (const barBounds of innerBars) {
                    const barBeats = barBounds.beats || [];
                    for (const beatBounds of barBeats) {
                        const bvb = beatBounds.visualBounds || beatBounds.bounds;
                        if (!bvb || bvb.x == null) continue;
                        const beat = beatBounds.beat;
                        // Only include non-rest beats for note matching
                        if (beat && (beat.isRest || beat.isEmpty)) continue;
                        beatCoords.push({
                            vb: {
                                x: bvb.x,
                                y: sysVb ? sysVb.y : bvb.y,
                                w: bvb.w,
                                h: sysVb ? sysVb.h : bvb.h,
                            },
                        });
                    }
                }
            }
        }

        if (notes.length > 0 && beatCoords.length > 0) {
            // Group notes into chord events (same time = 1 event)
            const sorted = [...notes].sort((a, b) => a.start - b.start);
            const noteEvents = [];
            let prev = -999;
            for (const n of sorted) {
                if (n.start - prev > 0.03) {
                    noteEvents.push({ startSec: n.start, endSec: n.end || n.start + 0.3 });
                }
                prev = n.start;
            }

            const mapLen = Math.min(noteEvents.length, beatCoords.length);
            if (mapLen > 0) {
                const map = [];
                for (let i = 0; i < mapLen; i++) {
                    const startMs = noteEvents[i].startSec * 1000;
                    const endMs = i + 1 < noteEvents.length
                        ? noteEvents[i + 1].startSec * 1000
                        : noteEvents[i].endSec * 1000 + 500;
                    map.push({ startMs, endMs, vb: beatCoords[i].vb });
                }
                beatMapRef.current = map;
                console.log(`[TabView] BeatMap (note-fallback): ${map.length} entries`);
                return true;
            }
        }

        // ===================================================================
        // Strategy 3: BPM-based even distribution (last resort)
        // ===================================================================
        const map = [];
        let accMs = 0;
        const forEach = (list, cb) => {
            if (Array.isArray(list)) list.forEach(cb);
            else if (list?.items) list.items.forEach(cb);
            else if (list?.forEach) list.forEach(cb);
        };
        const masterBars = api.score.masterBars;
        const getBeatsPerBar = (idx) => {
            let arr = [];
            if (masterBars) forEach(masterBars, mb => arr.push(mb));
            return idx < arr.length ? (arr[idx].timeSignatureNumerator || 4) : 4;
        };
        const barCoords = [];
        let barCounter = 0;
        const seenBars = new Set();
        for (const system of systems) {
            const sgBars = system.bars || system.masterBars;
            if (!sgBars) continue;
            const sysVb = system.visualBounds;
            for (const sgBar of sgBars) {
                const barIndex = sgBar.index ?? sgBar.barIndex ?? barCounter;
                barCounter++;
                if (seenBars.has(barIndex)) continue;
                seenBars.add(barIndex);
                const barVb = sgBar.visualBounds;
                if (barVb && barVb.x != null) {
                    barCoords.push({
                        barIndex,
                        vb: { x: barVb.x, y: sysVb ? sysVb.y : barVb.y, w: barVb.w, h: sysVb ? sysVb.h : barVb.h },
                    });
                }
            }
        }
        barCoords.sort((a, b) => a.barIndex - b.barIndex);
        for (const bc of barCoords) {
            const bpb = getBeatsPerBar(bc.barIndex);
            const durMs = bpb * (60000 / bpm);
            map.push({ startMs: accMs, endMs: accMs + durMs, vb: bc.vb });
            accMs += durMs;
        }
        beatMapRef.current = map;
        console.log(`[TabView] BeatMap (bpm-fallback): ${map.length} entries`);
        return map.length > 0;
    };

    const buildTechniqueOverlay = (api) => {
        // 市販出版譜の標準に完全準拠するため、独自テキストオーバーレイは完全廃止
        return;
    };

    const buildChordOverlay = (api) => {
        // コードはAlphaTabが五線譜最上部に美しくネイティブ描画するため、重複HTMLオーバーレイは廃止
        return;
    };

    const findBeat = (audioMs) => {
        const map = beatMapRef.current;
        if (!map || !map.length) return null;
        if (audioMs < map[0].startMs - 50) return null; // 曲開始前の無音区間はカーソル非表示
        if (audioMs > map[map.length - 1].endMs + 1500) return null; // 曲終了後は非表示
        let lo = 0, hi = map.length - 1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (audioMs >= map[mid].startMs && audioMs < map[mid].endMs) return map[mid];
            if (audioMs < map[mid].startMs) hi = mid - 1;
            else lo = mid + 1;
        }
        return lo > 0 && lo < map.length ? map[lo - 1] : map[0];
    };

    // ============================================================
    // AlphaTab init — NO innerHTML manipulation on React refs
    // ============================================================
    useEffect(() => {
        if (!sessionId || !wrapperRef.current) return;

        // Only sessionId, apiBase, reloadKey trigger GP5 re-fetch.
        // capo/tuning changes go through handleRetune → retuneKey → parent remount.
        // transpose is handled in a separate effect (no GP5 re-fetch needed).
        const key = `${sessionId}_${reloadKey}_${Date.now()}`;
        console.log(`[TabView] init triggered — key=${key}`);
        initKeyRef.current = key;

        let onWrapperClick = null;
        let destroyed = false;
        boundsReadyRef.current = false;
        beatMapRef.current = [];

        // Clear technique overlay
        if (wrapperRef.current?.parentElement) {
            const old = wrapperRef.current.parentElement.querySelector('.tech-overlay');
            if (old) old.remove();
        }

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
                // GP5バイナリを取得 — cache-busting with unique timestamp
                let res;
                let useGp5 = true;
                const cacheBuster = `t=${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
                for (let attempt = 0; attempt < 3; attempt++) {
                    const url = `${apiBase}/result/${sessionId}/gp5?${cacheBuster}&attempt=${attempt}`;
                    console.log(`[TabView] Fetching GP5: ${url}`);
                    res = await fetch(url, {
                        cache: 'no-store',
                        headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' },
                    });
                    if (res.ok) break;
                    console.warn(`[TabView] GP5 attempt ${attempt + 1} failed (status=${res.status}), retrying...`);
                    await new Promise(r => setTimeout(r, 1500));
                }
                if (!res || !res.ok) {
                    console.warn("[TabView] GP5 not available, falling back to MusicXML");
                    useGp5 = false;
                    for (let attempt = 0; attempt < 3; attempt++) {
                        res = await fetch(`${apiBase}/result/${sessionId}/musicxml?${cacheBuster}&attempt=${attempt}`, {
                            cache: 'no-store',
                            headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' },
                        });
                        if (res.ok) break;
                        await new Promise(r => setTimeout(r, 1500));
                    }
                    if (!res.ok) throw new Error("Score not available");
                }

                const scoreData = useGp5
                    ? new Uint8Array(await res.arrayBuffer())
                    : new TextEncoder().encode(await res.text());
                if (destroyed) return;

                console.log(`[TabView] Loaded ${useGp5 ? 'GP5' : 'MusicXML'}: ${scoreData.length} bytes (key=${key})`);

                // ノートデータ + beatsデータを取得（カーソル同期用）
                try {
                    const [notesRes, beatsRes, chordsRes] = await Promise.all([
                        fetch(`${apiBase}/result/${sessionId}/notes?${cacheBuster}`, {
                            cache: 'no-store',
                            headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' }
                        }),
                        fetch(`${apiBase}/files/${sessionId}/beats.json?${cacheBuster}`, {
                            cache: 'no-store',
                            headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' }
                        }),
                        fetch(`${apiBase}/files/${sessionId}/chords.json?${cacheBuster}`, {
                            cache: 'no-store',
                            headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' }
                        }),
                    ]);
                    if (notesRes.ok) {
                        const notesData = await notesRes.json();
                        notesDataRef.current = notesData.notes || [];
                        setCanUndo(notesData.can_undo || false);
                        setCanRedo(notesData.can_redo || false);
                        console.log(`[TabView] Loaded ${notesDataRef.current.length} notes for cursor sync`);
                    }
                    if (beatsRes.ok) {
                        const beatsData = await beatsRes.json();
                        let beats = Array.isArray(beatsData) ? beatsData : (beatsData.beats || []);
                        if (beats.length > 0 && typeof beats[0] === 'object') beats = beats.map(b => b.time);
                        beatsDataRef.current = beats;
                        console.log(`[TabView] Loaded ${beats.length} beats for piecewise cursor sync`);
                    }
                    if (chordsRes && chordsRes.ok) {
                        const chordsData = await chordsRes.json();
                        chordsDataRef.current = Array.isArray(chordsData) ? chordsData : [];
                        console.log(`[TabView] Loaded ${chordsDataRef.current.length} chords`);
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

                // === タイトル・指番号非表示（五線譜上の数字や文字化けを防止） ===
                const NE = window.alphaTab.NotationElement;
                if (NE) {
                    settings.notation.elements.set(NE.ScoreTitle, false);
                    settings.notation.elements.set(NE.ScoreSubTitle, false);
                    settings.notation.elements.set(NE.ScoreArtist, false);
                    settings.notation.elements.set(NE.ScoreWordsAndMusic, false);
                    if (NE.Fingering !== undefined) settings.notation.elements.set(NE.Fingering, false);
                    if (NE.LeftHandTap !== undefined) settings.notation.elements.set(NE.LeftHandTap, false);
                }

                if (settings.display.resources) {
                    settings.display.resources.titleFont = new window.alphaTab.model.Font("Arial", 16, 1);
                    // すべての音符・TAB数字・弱音・第2声部を完全な濃い黒（100%不透明）で描画
                    if (window.alphaTab.model?.Color) {
                        const solidBlack = new window.alphaTab.model.Color(0, 0, 0, 255);
                        settings.display.resources.mainGlyphColor = solidBlack;
                        settings.display.resources.secondaryGlyphColor = solidBlack;
                        settings.display.resources.scoreInfoColor = solidBlack;
                        settings.display.resources.fretNumberColor = solidBlack;
                    }
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
                    try {
                        const s = score?.score || score;
                        if (s && s.title !== undefined) {
                            console.log('[TabView] scoreLoaded: overriding title from', JSON.stringify(s.title), 'to', JSON.stringify(songTitle || ''));
                            s.title = songTitle || '';
                            s.subTitle = '';
                            s.artist = songTitle ? 'SoloTab' : '';
                            s.words = '';
                            s.music = '';
                        }
                    } catch (e) {
                        console.warn('[TabView] scoreLoaded title override failed:', e);
                    }
                });

                // Note click is now handled via coordinate-based matching inside handleWrapperClick to ensure reliability.;

                let renderTimeoutTimer = setTimeout(() => {
                    if (!destroyed && !boundsReadyRef.current) {
                        console.warn("[TabView] Render timeout (3s) reached. Resetting loading state.");
                        setLoading(false);
                    }
                }, 3000);

                api.renderStarted.on(() => setLoading(true));
                api.postRenderFinished.on(() => {
                    if (destroyed) return;
                    if (renderTimeoutTimer) clearTimeout(renderTimeoutTimer);
                    setLoading(false);


                    // Build BeatMap with retries
                    const tryBuild = (attempt) => {
                        if (destroyed || boundsReadyRef.current) return;
                        const ok = buildBeatMap(api);
                        boundsReadyRef.current = ok;
                        if (ok) {
                            console.log("[TabView] BeatMap ready");
                            // 独自テキストオーバーレイ（波線や文字バッジ）は完全撤廃し、市販出版譜の美しいネイティブ記譜に一本化
                            // コードはAlphaTabが五線譜最上部に美しく1回のみネイティブ描画（二重表示を防止）
                            try {
                                buildAnchorOverlay(api);
                            } catch (e) { console.warn("[TabView] Anchor overlay:", e); }
                        } else if (attempt < 4) {
                            setTimeout(() => tryBuild(attempt + 1), [500, 1000, 2000, 3000][attempt]);
                        }
                    };
                    tryBuild(0);
                    if (containerRef.current) containerRef.current.scrollTop = 0;
                });
                api.error.on((e) => {
                    console.error("[AlphaTab Error]", e);
                    if (renderTimeoutTimer) clearTimeout(renderTimeoutTimer);
                    if (!destroyed) { setError("TAB表示エラー"); setLoading(false); }
                });

                // Transpose is handled in a separate effect below

                console.log(`[TabView] Loading score data into AlphaTab...`);
                // --- 空白部分クリック → 新規ノート追加UI ---
                const wrapper = wrapperRef.current;
                onWrapperClick = (e) => {
                    handleWrapperClick(e, api);
                };
                if (wrapper) {
                    wrapper.addEventListener('click', onWrapperClick);
                }

                window.alphaTabApi = api;
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
            if (wrapperRef.current && onWrapperClick) {
                wrapperRef.current.removeEventListener('click', onWrapperClick);
            }
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sessionId, apiBase, reloadKey]);

    // Separate effect for frontend-only transpose — no GP5 re-fetch
    useEffect(() => {
        const api = apiRef.current;
        if (!api || !api.score) return;
        try {
            const trackList = api.score.tracks?.items || api.score.tracks || [];
            const forEach = (list, cb) => {
                if (Array.isArray(list)) list.forEach(cb);
                else if (list?.forEach) list.forEach(cb);
            };
            forEach(trackList, (track) => {
                forEach(track.staves, (staff) => {
                    staff.transpositionPitch = transpose;
                });
            });
            console.log(`[TabView] Applied transpose=${transpose}`);
            api.render();
        } catch (e) {
            console.warn('[TabView] Transpose failed:', e);
        }
    }, [transpose]);

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
            
            // Apply tempo multiplier and offset
            const tOffset = syncOffsetRef.current / 1000.0;
            const tMult = tempoMultiplierRef.current;
            const virtualTimeSec = (timeRef.current * tMult) + tOffset;
            const ms = Math.max(0, virtualTimeSec * 1000);
            
            const nowPlaying = playingRef.current;

            // --- Metronome ---
            if (nowPlaying && metronomeEnabledRef.current && audioCtxRef.current) {
                const beats = beatsDataRef.current;
                if (beats && beats.length > 0) {
                    const ctx = audioCtxRef.current;
                    // If index is invalid or time went backwards/jumped, search for next beat
                    if (nextBeatIdxRef.current < 0 || nextBeatIdxRef.current >= beats.length || beats[Math.max(0, nextBeatIdxRef.current - 1)] > virtualTimeSec + 0.2) {
                        let idx = 0;
                        while (idx < beats.length && beats[idx] < virtualTimeSec) idx++;
                        nextBeatIdxRef.current = idx;
                        scheduledOscsRef.current.forEach(osc => { try { osc.stop(); } catch(e){} });
                        scheduledOscsRef.current = [];
                    }

                    // Look ahead 0.1s
                    while (nextBeatIdxRef.current < beats.length) {
                        const beatTime = beats[nextBeatIdxRef.current];
                        const timeUntilBeat = beatTime - virtualTimeSec;

                        if (timeUntilBeat > 0.1) break;

                        if (timeUntilBeat >= 0) {
                            const osc = ctx.createOscillator();
                            const gain = ctx.createGain();
                            osc.frequency.value = (nextBeatIdxRef.current % 4 === 0) ? 1000 : 800;
                            osc.connect(gain);
                            gain.connect(ctx.destination);
                            
                            const playTime = ctx.currentTime + (timeUntilBeat / tMult);
                            osc.start(playTime);
                            osc.stop(playTime + 0.05);
                            osc._endTime = playTime + 0.05;
                            scheduledOscsRef.current.push(osc);

                            gain.gain.setValueAtTime(0, playTime);
                            gain.gain.linearRampToValueAtTime(1, playTime + 0.005);
                            gain.gain.exponentialRampToValueAtTime(0.01, playTime + 0.05);
                        }
                        nextBeatIdxRef.current++;
                    }
                    // Clean up old oscillators from the array
                    scheduledOscsRef.current = scheduledOscsRef.current.filter(o => o._endTime > ctx.currentTime);
                }
            }

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

                    // Thin vertical line — interpolate position within beat
                    const progress = (beat.endMs > beat.startMs) 
                        ? Math.min(1, Math.max(0, (ms - beat.startMs) / (beat.endMs - beat.startMs)))
                        : 0;
                    const cursorX = x + progress * w;

                    cursor.style.display = "block";
                    cursor.style.left = `${cursorX}px`;
                    cursor.style.top = `${y}px`;
                    cursor.style.height = `${h}px`;
                    // width is fixed at 3px via inline style

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
            <ScoreToolbar 
                sessionId={sessionId} 
                apiBase={apiBase} 
                onPdfExport={handlePdfExport} 
                isExporting={isExporting} 
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
                {/* Custom cursor (thin vertical line — precise beat tracking) */}
                <div
                    ref={cursorRef}
                    style={{
                        position: "absolute", display: "none", pointerEvents: "none",
                        zIndex: 30, top: 0, left: 0,
                        width: "3px",
                        background: "rgba(239, 68, 68, 0.85)",
                        borderRadius: 2,
                        boxShadow: "0 0 6px rgba(239, 68, 68, 0.4)",
                        transition: "left 0.08s linear, top 0.05s ease",
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
                                                {/* 指選択とアンカー */}
                        <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 4, marginBottom: 4 }}>
                            <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
                                <span style={{ fontSize: 10, color: "#64748b", width: 20 }}>指</span>
                                {[1,2,3,4].map(f => (
                                    <button key={f}
                                        onClick={() => setEditNote(prev => ({ ...prev, editFinger: prev.editFinger === f ? 0 : f }))}
                                        style={{
                                            width: 22, height: 22, borderRadius: 4, border: "none",
                                            background: editNote.editFinger === f ? "#10b981" : "#334155",
                                            color: "white", fontWeight: 700, fontSize: 11,
                                            cursor: "pointer", transition: "all 0.15s",
                                        }}
                                    >{f}</button>
                                ))}
                            </div>
                            <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer", fontSize: 11, color: editNote.editAnchor ? "gold" : "#94a3b8" }}>
                                <input type="checkbox" checked={editNote.editAnchor || false} onChange={(e) => setEditNote(prev => ({...prev, editAnchor: e.target.checked}))} style={{ accentColor: "gold" }} />
                                🔒 Lock
                            </label>
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
                                            const isNew = editNote.noteIndex === -1;
                                            const url = isNew
                                                ? `${apiBase}/result/${sessionId}/notes`
                                                : `${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`;
                                            const method = isNew ? "POST" : "PATCH";
                                            // Standard tuning MIDI base per string: 1=E4(64), 2=B3(59), 3=G3(55), 4=D3(50), 5=A2(45), 6=E2(40)
                                            const stringMidi = [0, 64, 59, 55, 50, 45, 40];
                                            const bodyData = isNew
                                            ? { fret: newFret, string: newString, finger: editNote.editFinger, anchor: editNote.editAnchor, start: editNote.startTime, end: editNote.startTime + 0.25, pitch: (stringMidi[newString] || 64) + newFret }
                                            : { fret: newFret, string: newString, finger: editNote.editFinger, anchor: editNote.editAnchor, start_time: editNote.startTime, old_fret: editNote.fret };

                                            const res = await fetch(url, {
                                                method: method,
                                                headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify(bodyData),
                                            });
                                            if (res.ok) {
                                                console.log('[TabView] Note saved:', await res.json());
                                                setEditNote(null);
                                                await new Promise(r => setTimeout(r, 300));
                                                onNoteEdited?.();
                                            } else {
                                                console.error('[TabView] Save failed:', res.status);
                                            }
                                        } catch (err) { console.error("Save failed:", err); }
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
                                        const isNew = editNote.noteIndex === -1;
                                        const url = isNew
                                            ? `${apiBase}/result/${sessionId}/notes`
                                            : `${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`;
                                        const method = isNew ? "POST" : "PATCH";
                                        // Standard tuning MIDI base per string: 1=E4(64), 2=B3(59), 3=G3(55), 4=D3(50), 5=A2(45), 6=E2(40)
                                        const stringMidi = [0, 64, 59, 55, 50, 45, 40];
                                        const bodyData = isNew
                                            ? { fret: newFret, string: newString, finger: editNote.editFinger, anchor: editNote.editAnchor, start: editNote.startTime, end: editNote.startTime + 0.25, pitch: (stringMidi[newString] || 64) + newFret }
                                            : { fret: newFret, string: newString, finger: editNote.editFinger, anchor: editNote.editAnchor, start_time: editNote.startTime, old_fret: editNote.fret };

                                        const res = await fetch(url, {
                                            method: method,
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify(bodyData),
                                        });
                                        if (res.ok) {
                                            console.log('[TabView] Note saved:', await res.json());
                                            setEditNote(null);
                                            await new Promise(r => setTimeout(r, 300));
                                            onNoteEdited?.();
                                        } else {
                                            console.error('[TabView] Save failed:', res.status);
                                        }
                                    } catch (err) { console.error("Save failed:", err); }
                                    setEditSaving(false);
                                }}
                                disabled={editSaving}
                                style={{
                                    padding: "4px 10px", borderRadius: 6, border: "none",
                                    background: "#3b82f6", color: "white", fontWeight: 700,
                                    cursor: "pointer", fontSize: 13,
                                }}
                            >✓</button>
                            {editNote.noteIndex !== -1 && (
                                <button
                                    onClick={async () => {
                                        if (!confirm("このノートを削除しますか？")) return;
                                        setEditSaving(true);
                                        try {
                                            const res = await fetch(`${apiBase}/result/${sessionId}/notes/${editNote.noteIndex}`, {
                                                method: "PATCH",
                                                headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({ delete: true, start_time: editNote.startTime, string: editNote.string, old_fret: editNote.fret }),
                                            });
                                            if (res.ok) {
                                                setEditNote(null);
                                                await new Promise(r => setTimeout(r, 300));
                                                onNoteEdited?.();
                                            }
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
                            )}
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
                                {/* Undo / Redo */}
                <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                    <div
                        style={{
                            padding: "8px 12px", borderRadius: 20, cursor: canUndo && !loading ? "pointer" : "default",
                            background: canUndo ? "#3b82f6" : "#475569", color: canUndo ? "white" : "#94a3b8",
                            fontSize: 12, fontWeight: 700, boxShadow: "0 4px 16px rgba(0,0,0,0.3)",
                            transition: "all 0.2s", userSelect: "none", opacity: loading ? 0.5 : 1
                        }}
                        onClick={handleUndo}
                    >
                        ↩ Undo
                    </div>
                    <div
                        style={{
                            padding: "8px 12px", borderRadius: 20, cursor: canRedo && !loading ? "pointer" : "default",
                            background: canRedo ? "#3b82f6" : "#475569", color: canRedo ? "white" : "#94a3b8",
                            fontSize: 12, fontWeight: 700, boxShadow: "0 4px 16px rgba(0,0,0,0.3)",
                            transition: "all 0.2s", userSelect: "none", opacity: loading ? 0.5 : 1
                        }}
                        onClick={handleRedo}
                    >
                        ↪ Redo
                    </div>
                </div>

                {/* Reset Anchors */}
                <div
                    style={{
                        padding: "8px 16px", borderRadius: 20, cursor: "pointer",
                        background: "#ef4444", color: "white", fontSize: 12, fontWeight: 700,
                        boxShadow: "0 4px 16px rgba(0,0,0,0.3)", transition: "all 0.2s", userSelect: "none",
                    }}
                    onClick={async () => {
                        if (!confirm("すべてのアンカー（固定された運指）を解除し、AIの推論をリセットしますか？")) return;
                        try {
                            setLoading(true);
                            const res = await fetch(`${apiBase}/result/${sessionId}/anchors/reset`, { method: "POST" });
                            if (res.ok) {
                                await new Promise(r => setTimeout(r, 500));
                                onNoteEdited?.();
                            }
                        } catch (e) {
                            console.error(e);
                        } finally {
                            setLoading(false);
                        }
                    }}
                >
                    🔓 RESET ANCHORS
                </div>
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

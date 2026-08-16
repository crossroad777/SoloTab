import re

with open("frontend/src/components/TabView.jsx", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add editAnchor and editFinger to handleWrapperClick
content = content.replace(
"""                setEditNote({
                    noteIndex: backendIdx,
                    fret: note.fret,
                    string: backendString,""",
"""                setEditNote({
                    noteIndex: backendIdx,
                    fret: note.fret,
                    string: backendString,
                    editFinger: matchedNote?.left_hand_finger || 0,
                    editAnchor: matchedNote?._is_anchor || false,"""
)

# 2. Add buildAnchorOverlay function
build_anchor_overlay = """    const buildAnchorOverlay = (api) => {
        if (!wrapperRef.current) return;
        const parent = wrapperRef.current.parentElement;
        if (!parent) return;

        const old = parent.querySelector('.anchor-overlay');
        if (old) old.remove();

        const notes  = notesDataRef.current;
        const beatMap = beatMapRef.current;
        if (!notes.length || !beatMap.length) return;

        const overlay = document.createElement('div');
        overlay.className = 'anchor-overlay';
        overlay.style.position = 'absolute';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.pointerEvents = 'none';
        overlay.style.zIndex = '15';

        const lookup = api?.renderer?.boundsLookup;
        if (!lookup) return;
        const margin = 2;

        const groups = lookup.staffSystems || lookup.staveGroups || [];
        for (const sys of groups) {
            const bars = sys.bars || sys.masterBars || [];
            for (const bar of bars) {
                const barBounds = bar.bars || [];
                for (const bb of barBounds) {
                    const beats = bb.beats || [];
                    for (const beatBounds of beats) {
                        const notesBounds = beatBounds.notes || [];
                        for (const nb of notesBounds) {
                            if (!nb.note) continue;
                            const scoreNote = nb.note;
                            
                            const beat = scoreNote.beat;
                            const ticksPerBeat = 960;
                            const bpm = (typeof api.score?.tempo === 'object' && api.score?.tempo !== null)
                                ? (api.score.tempo.value || 120)
                                : (api.score?.tempo || 120);
                            const approxTimeSec = (beat.absolutePlaybackStart / ticksPerBeat) * (60 / bpm);
                            const backendString = 7 - scoreNote.string;

                            let matchedBackendNote = null;
                            let bestDist = Infinity;
                            for (let i = 0; i < notes.length; i++) {
                                const bn = notes[i];
                                const startVal = bn.start_time ?? bn.start ?? 0;
                                if (Number(bn.string) === backendString && Number(bn.fret) === Number(scoreNote.fret)) {
                                    const d = Math.abs(startVal - approxTimeSec);
                                    if (d < bestDist) { bestDist = d; matchedBackendNote = bn; }
                                }
                            }
                            
                            if (matchedBackendNote && matchedBackendNote._is_anchor) {
                                const bounds = nb.noteHeadBounds || beatBounds.visualBounds;
                                if (bounds) {
                                    const { x, y, w, h } = bounds;
                                    const el = document.createElement('div');
                                    el.innerText = '🔒';
                                    el.style.position = 'absolute';
                                    el.style.left = `${x}px`;
                                    el.style.top = `${y - 12}px`;
                                    el.style.fontSize = '12px';
                                    el.style.color = 'gold';
                                    el.style.textShadow = '0 0 2px black';
                                    overlay.appendChild(el);
                                    
                                    const bg = document.createElement('div');
                                    bg.style.position = 'absolute';
                                    bg.style.left = `${x - margin}px`;
                                    bg.style.top = `${y - margin}px`;
                                    bg.style.width = `${w + margin*2}px`;
                                    bg.style.height = `${h + margin*2}px`;
                                    bg.style.background = 'rgba(255, 215, 0, 0.2)';
                                    bg.style.border = '1px solid rgba(255, 215, 0, 0.5)';
                                    bg.style.borderRadius = '4px';
                                    bg.style.zIndex = '-1';
                                    overlay.appendChild(bg);
                                }
                            }
                        }
                    }
                }
            }
        }
        parent.appendChild(overlay);
    };

    const buildChordOverlay"""

content = content.replace("    const buildChordOverlay", build_anchor_overlay)

# 3. Call buildAnchorOverlay
call_anchor_overlay = """                            try {
                                buildChordOverlay(api);
                            } catch (e) { console.warn("[TabView] Chord overlay:", e); }
                            try {
                                buildAnchorOverlay(api);
                            } catch (e) { console.warn("[TabView] Anchor overlay:", e); }"""

content = content.replace(
"""                            try {
                                buildChordOverlay(api);
                            } catch (e) { console.warn("[TabView] Chord overlay:", e); }""",
call_anchor_overlay
)

# 4. Modify edit modal UI and fetch logic
# We need to replace the edit modal part. I will use regex to find the modal structure.
edit_modal_fetch = """const bodyData = isNew
                                            ? { fret: newFret, string: newString, finger: editNote.editFinger, anchor: editNote.editAnchor, start: editNote.startTime, end: editNote.startTime + 0.25, pitch: (stringMidi[newString] || 64) + newFret }
                                            : { fret: newFret, string: newString, finger: editNote.editFinger, anchor: editNote.editAnchor, start_time: editNote.startTime, old_fret: editNote.fret };"""
content = re.sub(
r"const bodyData = isNew\s*\?\s*\{\s*fret:\s*newFret,\s*string:\s*newString,\s*start:\s*editNote\.startTime,\s*end:\s*editNote\.startTime\s*\+\s*0\.25,\s*pitch:\s*\(stringMidi\[newString\]\s*\|\|\s*64\)\s*\+\s*newFret\s*\}\s*:\s*\{\s*fret:\s*newFret,\s*string:\s*newString,\s*start_time:\s*editNote\.startTime,\s*old_fret:\s*editNote\.fret\s*\};",
edit_modal_fetch,
content
)

# Insert Finger and Anchor UI
finger_ui = """                        {/* 指選択とアンカー */}
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
                        {/* フレット入力 + 保存 */}"""

content = content.replace("{/* フレット入力 + 保存 */}", finger_ui)

# 5. Add Reset All Anchors button to the zoom controls (bottom right)
reset_btn = """                {/* Reset Anchors */}
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
                {/* ズームコントロール */}"""

content = content.replace("{/* ズームコントロール */}", reset_btn)

with open("frontend/src/components/TabView.jsx", "w", encoding="utf-8") as f:
    f.write(content)

print("Patched TabView.jsx successfully!")

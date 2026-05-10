import React, { useState, useEffect, useCallback, useRef } from "react";

/**
 * ScoreToolbar — PowerTab互換エディタツールバー V3
 * コンパクト1行 + 展開式詳細パネル
 */

const TECHNIQUES = [
  { key: "hammer_on", label: "H", title: "ハンマリング" },
  { key: "pull_off", label: "P", title: "プリング" },
  { key: "slide_up", label: "S↑", title: "スライドアップ" },
  { key: "slide_down", label: "S↓", title: "スライドダウン" },
  { key: "bend", label: "B", title: "ベンド" },
  { key: "vibrato", label: "~", title: "ビブラート" },
  { key: "palm_mute", label: "PM", title: "パームミュート" },
  { key: "harmonic", label: "◇", title: "ハーモニクス" },
  { key: "ghost_note", label: "x", title: "ゴーストノート" },
  { key: "let_ring", label: "LR", title: "レットリング" },
  { key: "trill", label: "Tr", title: "トリル" },
];

const DURATIONS = [
  { v: "whole", l: "𝅝" }, { v: "half", l: "𝅗𝅥" },
  { v: "quarter", l: "♩" }, { v: "eighth", l: "♪" },
  { v: "16th", l: "𝅘𝅥𝅯" },
];

const DYNAMICS = ["pp", "p", "mp", "mf", "f", "ff"];
const BARLINES = [
  { value: "normal", label: "─" }, { value: "double", label: "║" },
  { value: "final", label: "▐" }, { value: "repeat_start", label: "𝄆" },
  { value: "repeat_end", label: "𝄇" },
];

const B = {
  padding: "3px 6px", border: "1px solid #d1d5db", borderRadius: 3,
  background: "#fff", cursor: "pointer", fontSize: 11, fontWeight: 600,
  transition: "all 0.1s", minWidth: 22, textAlign: "center", lineHeight: "16px",
};
const BA = { ...B, background: "#3b82f6", color: "#fff", borderColor: "#3b82f6" };
const S = { display: "flex", alignItems: "center", gap: 3, padding: "0 6px", borderRight: "1px solid #e5e7eb" };
const L = { fontSize: 9, color: "#6b7280", fontWeight: 700, marginRight: 2, whiteSpace: "nowrap" };

const ScoreToolbar = ({ sessionId, apiBase, onScoreUpdate, score }) => {
  const [selectedBar, setSelectedBar] = useState(0);
  const [selectedBeat, setSelectedBeat] = useState(-1);
  const [selectedNote, setSelectedNote] = useState(-1);
  const [barProps, setBarProps] = useState(null);
  const [expanded, setExpanded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [clipboard, setClipboard] = useState(null);
  const undoStack = useRef([]);
  const redoStack = useRef([]);

  useEffect(() => {
    if (score?.bars?.[selectedBar]) setBarProps({ ...score.bars[selectedBar] });
    setSelectedBeat(-1); setSelectedNote(-1);
  }, [score, selectedBar]);

  const totalBars = score?.bars?.length || 0;
  const curBar = score?.bars?.[selectedBar];
  const curBeat = curBar?.beats?.[selectedBeat];
  const curNote = curBeat?.notes?.[selectedNote];

  const pushUndo = useCallback(() => {
    if (score) { undoStack.current.push(JSON.stringify(score)); redoStack.current = []; }
    if (undoStack.current.length > 30) undoStack.current.shift();
  }, [score]);

  const apiCall = useCallback(async (method, path, body = null) => {
    setSaving(true);
    try {
      const opts = { method, headers: { "Content-Type": "application/json" } };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(`${apiBase}${path}`, opts);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (onScoreUpdate) onScoreUpdate();
      return data;
    } catch (e) { console.error("API error:", e); }
    finally { setSaving(false); }
  }, [apiBase, onScoreUpdate]);

  const undo = useCallback(async () => {
    if (!undoStack.current.length) return;
    redoStack.current.push(JSON.stringify(score));
    await apiCall("PUT", `/result/${sessionId}/score`, JSON.parse(undoStack.current.pop()));
  }, [score, sessionId]);
  const redo = useCallback(async () => {
    if (!redoStack.current.length) return;
    undoStack.current.push(JSON.stringify(score));
    await apiCall("PUT", `/result/${sessionId}/score`, JSON.parse(redoStack.current.pop()));
  }, [score, sessionId]);

  const addBar = () => { pushUndo(); apiCall("POST", `/result/${sessionId}/score/bars`, { after_bar: selectedBar }); };
  const deleteBar = () => { if (totalBars <= 1) return; pushUndo(); apiCall("DELETE", `/result/${sessionId}/score/bars/${selectedBar}`); };
  const updateBarProp = (k, v) => { pushUndo(); apiCall("PATCH", `/result/${sessionId}/score/bars/${selectedBar}`, { [k]: v }); };
  const changeDuration = (d) => { if (selectedBeat < 0) return; pushUndo(); apiCall("PATCH", `/result/${sessionId}/score/bars/${selectedBar}/beats/${selectedBeat}`, { duration: d }); };
  const toggleTriplet = () => { if (selectedBeat < 0 || !curBeat) return; pushUndo(); apiCall("PATCH", `/result/${sessionId}/score/bars/${selectedBar}/beats/${selectedBeat}`, { triplet: !curBeat.triplet }); };
  const toggleDotted = () => { if (selectedBeat < 0 || !curBeat) return; pushUndo(); apiCall("PATCH", `/result/${sessionId}/score/bars/${selectedBar}/beats/${selectedBeat}`, { dotted: !curBeat.dotted }); };
  const toggleTechnique = (tk) => {
    if (selectedBeat < 0 || selectedNote < 0 || !curNote) return; pushUndo();
    const cur = curNote.techniques?.[tk];
    let nv; if (tk === "bend") nv = cur ? null : { type: "full", value: 1.0 }; else if (tk === "harmonic") nv = cur ? null : "natural"; else if (tk === "trill") nv = cur ? null : { fret: (curNote.fret||0)+2 }; else nv = !cur;
    apiCall("PATCH", `/result/${sessionId}/score/bars/${selectedBar}/beats/${selectedBeat}/notes/${selectedNote}`, { techniques: { [tk]: nv } });
  };
  const copyBar = () => { if (curBar) setClipboard(JSON.parse(JSON.stringify(curBar))); };
  const pasteBar = () => { if (!clipboard || !score) return; pushUndo(); const ns = JSON.parse(JSON.stringify(score)); ns.bars.splice(selectedBar+1,0,{...clipboard}); ns.bars.forEach((b,i)=>b.bar_number=i+1); apiCall("PUT",`/result/${sessionId}/score`,ns); };

  useEffect(() => {
    const h = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.ctrlKey && e.key === "z") { e.preventDefault(); undo(); }
      if (e.ctrlKey && e.key === "y") { e.preventDefault(); redo(); }
      if (e.ctrlKey && e.key === "c") { e.preventDefault(); copyBar(); }
      if (e.ctrlKey && e.key === "v") { e.preventDefault(); pasteBar(); }
      if (e.key === "ArrowLeft") setSelectedBar(b => Math.max(0, b - 1));
      if (e.key === "ArrowRight") setSelectedBar(b => Math.min(totalBars - 1, b + 1));
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [undo, redo, totalBars]);

  if (!sessionId) return null;
  if (!score) return (
    <div style={{ padding: "6px 12px", borderBottom: "1px solid #e5e7eb", background: "#fafbfc", fontSize: 11, color: "#9ca3af" }}>
      スコアデータを読み込み中...
    </div>
  );

  return (
    <div style={{ borderBottom: "2px solid #e5e7eb", background: "#fafbfc", userSelect: "none" }}>
      {/* メインバー: 1行にすべて収める */}
      <div style={{ display: "flex", alignItems: "center", flexWrap: "nowrap", padding: "3px 6px", overflowX: "auto", gap: 0 }}>
        {/* 小節ナビ */}
        <div style={S}>
          <button style={B} onClick={() => setSelectedBar(Math.max(0, selectedBar - 1))} disabled={selectedBar <= 0}>◀</button>
          <span style={{ fontSize: 11, fontWeight: 700, minWidth: 40, textAlign: "center" }}>{selectedBar + 1}/{totalBars}</span>
          <button style={B} onClick={() => setSelectedBar(Math.min(totalBars - 1, selectedBar + 1))} disabled={selectedBar >= totalBars - 1}>▶</button>
        </div>
        {/* 操作 */}
        <div style={S}>
          <button style={B} onClick={addBar} title="小節追加">＋</button>
          <button style={{...B,color:"#ef4444"}} onClick={deleteBar} title="小節削除">−</button>
          <button style={B} onClick={undo} title="Ctrl+Z">↩</button>
          <button style={B} onClick={redo} title="Ctrl+Y">↪</button>
          <button style={B} onClick={copyBar} title="Ctrl+C">📋</button>
          <button style={B} onClick={pasteBar} title="Ctrl+V" disabled={!clipboard}>📌</button>
        </div>
        {/* 音価 */}
        <div style={S}>
          {DURATIONS.map(d => (
            <button key={d.v} style={curBeat?.duration===d.v?BA:B} onClick={()=>changeDuration(d.v)} title={d.v}>{d.l}</button>
          ))}
          <button style={curBeat?.dotted?BA:B} onClick={toggleDotted} title="付点">.</button>
          <button style={curBeat?.triplet?BA:B} onClick={toggleTriplet} title="3連符">3</button>
        </div>
        {/* テクニック */}
        <div style={S}>
          {TECHNIQUES.map(t => {
            const active = curNote?.techniques?.[t.key] && curNote.techniques[t.key] !== false && curNote.techniques[t.key] !== null;
            return <button key={t.key} style={active?BA:(selectedNote>=0?B:{...B,opacity:0.4})} onClick={()=>toggleTechnique(t.key)} title={t.title} disabled={selectedNote<0}>{t.label}</button>;
          })}
        </div>
        {/* 展開ボタン */}
        <div style={{...S, borderRight:"none"}}>
          <button style={expanded?BA:B} onClick={()=>setExpanded(!expanded)} title="詳細設定">
            {expanded ? "▲" : "▼"}
          </button>
          {saving && <span style={{fontSize:10,color:"#3b82f6",marginLeft:4}}>⏳</span>}
        </div>
      </div>

      {/* 展開パネル */}
      {expanded && (
        <div style={{ padding: "4px 8px", borderTop: "1px solid #e5e7eb", background: "#f8fafc",
          display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
          <div style={S}>
            <span style={L}>拍子</span>
            <select style={{...B,minWidth:45}} value={barProps?.time_signature||"4/4"} onChange={e=>updateBarProp("time_signature",e.target.value)}>
              {["2/4","3/4","4/4","5/4","6/8","7/8","12/8"].map(t=><option key={t}>{t}</option>)}
            </select>
          </div>
          <div style={S}>
            <span style={L}>BPM</span>
            <input type="number" style={{...B,width:45}} value={barProps?.tempo||120} min={30} max={300}
              onBlur={e=>updateBarProp("tempo",parseInt(e.target.value)||120)}
              onChange={e=>setBarProps(p=>({...p,tempo:e.target.value}))} />
          </div>
          <div style={S}>
            <span style={L}>線</span>
            {BARLINES.map(b=><button key={b.value} style={barProps?.barline_end===b.value?BA:B} onClick={()=>updateBarProp("barline_end",b.value)} title={b.label}>{b.label}</button>)}
          </div>
          <div style={S}>
            <span style={L}>強弱</span>
            {DYNAMICS.map(d=><button key={d} style={barProps?.dynamic===d?BA:B} onClick={()=>updateBarProp("dynamic",barProps?.dynamic===d?null:d)}>{d}</button>)}
          </div>
          <div style={S}>
            <span style={L}>コード</span>
            <input type="text" placeholder="Am7" style={{...B,width:50}} value={barProps?.chord_text||""}
              onBlur={e=>updateBarProp("chord_text",e.target.value||null)}
              onChange={e=>setBarProps(p=>({...p,chord_text:e.target.value}))} />
          </div>
          <div style={S}>
            <span style={L}>区間</span>
            <input type="text" placeholder="A" style={{...B,width:25,textAlign:"center"}} maxLength={3}
              value={barProps?.rehearsal_sign||""}
              onBlur={e=>updateBarProp("rehearsal_sign",e.target.value||null)}
              onChange={e=>setBarProps(p=>({...p,rehearsal_sign:e.target.value}))} />
          </div>
          <div style={{...S,borderRight:"none"}}>
            <span style={L}>反復</span>
            <select style={{...B,minWidth:50}} value={barProps?.direction||""} onChange={e=>updateBarProp("direction",e.target.value||null)}>
              <option value="">-</option>
              {["coda","segno","fine","da_capo","dal_segno"].map(d=><option key={d} value={d}>{d.replace(/_/g," ")}</option>)}
            </select>
          </div>
        </div>
      )}

      {/* ビートエディタ（展開時のみ） */}
      {expanded && curBar?.beats?.length > 0 && (
        <div style={{ padding: "3px 8px", borderTop: "1px solid #e5e7eb", background: "#f0f9ff", overflowX: "auto" }}>
          <div style={{ display: "flex", gap: 2, alignItems: "flex-start" }}>
            <span style={{...L,paddingTop:4}}>ビート:</span>
            {curBar.beats.map((beat, bIdx) => (
              <div key={bIdx}
                onClick={() => { setSelectedBeat(bIdx); setSelectedNote(beat.notes?.length ? 0 : -1); }}
                style={{
                  display: "inline-flex", flexDirection: "column", alignItems: "center",
                  padding: "2px 4px", borderRadius: 3, cursor: "pointer",
                  background: bIdx===selectedBeat ? "#dbeafe" : (beat.rest?"#fef2f2":"#f0fdf4"),
                  border: `1.5px solid ${bIdx===selectedBeat?"#3b82f6":(beat.rest?"#fca5a5":"#86efac")}`,
                  fontSize: 10, minWidth: 30, position: "relative",
                }}>
                <div style={{fontSize:8,color:"#9ca3af"}}>{beat.duration}{beat.triplet?"×3":""}</div>
                {beat.rest ? <div style={{color:"#ef4444",fontWeight:600,fontSize:9}}>休</div> :
                  beat.notes?.map((n, nIdx) => (
                    <div key={nIdx}
                      onClick={e=>{ e.stopPropagation(); setSelectedBeat(bIdx); setSelectedNote(nIdx); }}
                      style={{ fontSize:10, fontWeight:600, cursor:"pointer", padding:"0 1px", borderRadius:2,
                        background: bIdx===selectedBeat&&nIdx===selectedNote?"#93c5fd":"transparent" }}>
                      <span style={{color:"#3b82f6"}}>{n.string}</span>:{n.fret}
                    </div>
                  ))
                }
                {bIdx===selectedBeat && (
                  <button onClick={e=>{ e.stopPropagation(); pushUndo(); apiCall("DELETE",`/result/${sessionId}/score/bars/${selectedBar}/beats/${bIdx}`); setSelectedBeat(-1); }}
                    style={{position:"absolute",top:-5,right:-5,width:12,height:12,borderRadius:"50%",border:"none",background:"#ef4444",color:"#fff",fontSize:8,cursor:"pointer",lineHeight:"12px",padding:0}}>×</button>
                )}
              </div>
            ))}
            <button style={{...B,padding:"4px 8px",alignSelf:"center"}}
              onClick={()=>{ pushUndo(); const lp=curBar.beats.length>0?Math.max(...curBar.beats.map(b=>b.position))+12:0; apiCall("POST",`/result/${sessionId}/score/bars/${selectedBar}/beats`,{position:lp,duration:"quarter",string:1,fret:0,pitch:64}); }}
              title="ビート追加">＋</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ScoreToolbar;

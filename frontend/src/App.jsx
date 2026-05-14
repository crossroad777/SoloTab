import React, { useState, useEffect, useRef } from "react";
import { UploadCloud, Play, Pause, Square, History, Music, ChevronRight, Sun, Moon, Home, Download, Printer, Activity, Repeat, Headphones, VolumeX, Timer, MoreVertical, Edit2, ChevronUp, SkipBack } from "lucide-react";
import { TabView } from "./components/TabView";

const API_BASE = import.meta.env.VITE_API_URL !== undefined ? import.meta.env.VITE_API_URL : "http://localhost:8001";
const STATUS = { IDLE: "idle", UPLOADING: "uploading", PROCESSING: "processing", COMPLETED: "completed", FAILED: "failed" };

export default function SoloTabApp() {
  const [status, setStatus] = useState(STATUS.IDLE);
  const [progressMsg, setProgressMsg] = useState("");
  const [stepsDone, setStepsDone] = useState(0);
  const [session, setSession] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [noiseGate, setNoiseGate] = useState(0.20);
  const [loopA, setLoopA] = useState(null);
  const [loopB, setLoopB] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [ytUrl, setYtUrl] = useState("");
  const [retuneKey, setRetuneKey] = useState(0);
  const [retuning, setRetuning] = useState(false);
  const [transpose, setTranspose] = useState(0);
  const [capo, setCapo] = useState(0);
  const [history, setHistory] = useState([]);
  const [toast, setToast] = useState(null);
  const [soloGuitar, setSoloGuitar] = useState(true);
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('solotab-theme') || 'dark'; } catch { return 'dark'; }
  });

  const audioRef = useRef(null);
  const sseRef = useRef(null);
  const fileInputRef = useRef(null);
  const scrollTimerRef = useRef(null);
  const scrollStartRef = useRef(null);
  const [scrollOnly, setScrollOnly] = useState(false);
  // alphaTabApiRef removed (AlphaTab排除済み)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('solotab-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  // Fetch history
  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      if (res.ok) setHistory(await res.json());
    } catch (e) { console.error("History fetch:", e); }
  };

  useEffect(() => { if (status === STATUS.IDLE) fetchHistory(); }, [status]);

  // 起動時に履歴を取得（トップページ表示）
  useEffect(() => {
    fetchHistory();
  }, []);

  // Audio sync & loop logic
  useEffect(() => {
    let anim;
    let lastTime = 0;
    const tick = () => {
      if (audioRef.current) {
        const t = audioRef.current.currentTime;
        
        // Handle A-B Looping
        if (loopA !== null && loopB !== null && loopA < loopB) {
          if (t >= loopB) {
            audioRef.current.currentTime = loopA;
            lastTime = loopA;
            setCurrentTime(loopA);
            anim = requestAnimationFrame(tick);
            return;
          }
        }

        if (Math.abs(t - lastTime) > 0.03) {
          lastTime = t;
          setCurrentTime(t);
        }
        anim = requestAnimationFrame(tick);
      }
    };
    if (isPlaying) {
      if (audioRef.current) audioRef.current.playbackRate = speed;
      anim = requestAnimationFrame(tick);
    }
    return () => cancelAnimationFrame(anim);
  }, [isPlaying, loopA, loopB, speed]);
  
  // Speed application outside of play loop
  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = speed;
  }, [speed]);

  const restoreSession = async (sid) => {
    setStatus(STATUS.PROCESSING);
    setProgressMsg("セッション復元中...");
    try {
      const res = await fetch(`${API_BASE}/result/${sid}`);
      if (!res.ok) throw new Error("Not found");
      const result = await res.json();
      setSession({
        id: sid,
        fileName: result.filename || "Restored",
        bpm: result.bpm,
        totalNotes: result.total_notes,
        tuning: result.tuning,
        detectedKey: result.key || null,
        detectedCapo: result.capo || 0,
        audioUrl: `${API_BASE}/files/${sid}/converted.wav`,
      });
      if (result.capo > 0) setCapo(result.capo);
      if (result.noise_gate !== null && result.noise_gate !== undefined) setNoiseGate(result.noise_gate);
      setStatus(STATUS.COMPLETED);
    } catch {
      setStatus(STATUS.IDLE);
      localStorage.removeItem('solotab-last-session');
      fetchHistory();
    }
  };

  // SSE
  const startStatusStream = (sid) => {
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
    const es = new EventSource(`${API_BASE}/status/${sid}/stream`);
    sseRef.current = es;

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setProgressMsg(data.progress || "解析中...");
        if (typeof data.steps_done === 'number') setStepsDone(data.steps_done);
        if (data.filename) setSession(prev => prev ? { ...prev, fileName: data.filename } : prev);

        if (data.status === "completed") {
          es.close(); sseRef.current = null;
          handleCompleted(sid);
        } else if (data.status === "failed" || data.status === "not_found") {
          es.close(); sseRef.current = null;
          setProgressMsg(data.error || "解析に失敗しました");
          setStatus(STATUS.FAILED);
        }
      } catch (e) { console.error("[SSE] Parse:", e); }
    };

    es.onerror = () => {
      es.close(); sseRef.current = null;
      // fallback polling
      const poll = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/status/${sid}`);
          const data = await res.json();
          setProgressMsg(data.progress || "解析中...");
          if (typeof data.steps_done === 'number') setStepsDone(data.steps_done);
          if (data.status === "completed") { clearInterval(poll); handleCompleted(sid); }
          else if (data.status === "failed") { clearInterval(poll); setStatus(STATUS.FAILED); setProgressMsg(data.error || "エラー"); }
        } catch { /* polling error, retry next interval */ }
      }, 2000);
    };
  };

  const handleCompleted = async (sid) => {
    try {
      const res = await fetch(`${API_BASE}/result/${sid}`);
      const result = await res.json();
      setSession(prev => ({
        ...prev,
        bpm: result.bpm,
        totalNotes: result.total_notes,
        tuning: result.tuning,
        detectedKey: result.key || null,
        detectedCapo: result.capo || 0,
        fileName: result.filename || prev?.fileName,
        audioUrl: prev?.audioUrl || `${API_BASE}/files/${sid}/converted.wav`,
      }));
      if (result.capo > 0) setCapo(result.capo);
      setStatus(STATUS.COMPLETED);
    } catch {
      setStatus(STATUS.FAILED);
      setProgressMsg("結果取得に失敗");
    }
  };

  // Upload
  const handleUpload = async (file) => {
    if (!file) return;
    const isAudio = file.name.match(/\.(mp3|wav|m4a|flac)$/i);
    if (!isAudio) {
      // Try reading as text for YouTube URL (shortcut files etc.)
      if (file.size > 256000) {
        setStatus(STATUS.FAILED);
        setProgressMsg("サポートされていない形式です (MP3, WAV, M4A, FLAC)");
        return;
      }
      const reader = new FileReader();
      reader.onload = (ev) => {
        const content = ev.target.result;
        const ytMatch = content.match(/(https?:\/\/(?:www\.|music\.|m\.)?youtube\.com\/watch\?v=[^\s"']+(?:&[^\s"']+)?)|(https?:\/\/youtu\.be\/[^\s?]+(?:\?[^\s"']+)?)|(https?:\/\/(?:www\.|music\.)?youtube\.com\/shorts\/[^\s"']+)/i);
        if (ytMatch) { handleYouTubeUpload(ytMatch[0].trim()); }
        else { setStatus(STATUS.FAILED); setProgressMsg("サポートされていない形式です。音声ファイルまたはYouTubeリンクをドロップしてください。"); }
      };
      reader.readAsText(file);
      return;
    }
    setStatus(STATUS.UPLOADING);
    setProgressMsg("アップロード中...");
    setStepsDone(0);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("skip_demucs", soloGuitar);
    formData.append("fast_moe", "true");
    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      setSession({ id: data.session_id, fileName: file.name, audioUrl: `${API_BASE}${data.audio_url}` });
      setStatus(STATUS.PROCESSING);
      setStepsDone(1); // アップロード完了 = 20%
      setProgressMsg("ビート検出中...");
      startStatusStream(data.session_id);
    } catch (err) {
      setStatus(STATUS.FAILED);
      setProgressMsg(err.message || "アップロードに失敗しました");
    }
  };

  // YouTube Upload
  const handleYouTubeUpload = async (urlToUse = ytUrl) => {
    if (!urlToUse.trim()) return;
    setStatus(STATUS.PROCESSING);
    setStepsDone(0);
    setProgressMsg("YouTube音声を解析中...");
    try {
      const res = await fetch(`${API_BASE}/upload/youtube`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: urlToUse.trim() }),
      });
      if (!res.ok) throw new Error("YouTube upload failed");
      const data = await res.json();
      setSession({ id: data.session_id, fileName: "YouTube Video" });
      setYtUrl("");
      startStatusStream(data.session_id);
    } catch {
      setStatus(STATUS.FAILED);
      setProgressMsg("YouTube解析に失敗しました。URLを確認してください。");
    }
  };

  // D&D handlers on root element
  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    // Check for YouTube URL in dropped text
    const dt = e.dataTransfer;
    let droppedText = dt.getData("text/plain") || dt.getData("text/uri-list") || dt.getData("text");
    if (droppedText && droppedText.includes("\n")) droppedText = droppedText.split("\n")[0].trim();
    if (droppedText && (droppedText.includes("youtube.com") || droppedText.includes("youtu.be"))) {
      handleYouTubeUpload(droppedText.trim());
      return;
    }
    // Check for files
    if (dt.files?.[0]) {
      handleUpload(dt.files[0]);
      return;
    }
  };

  const handleSeek = (time) => {
    if (audioRef.current) audioRef.current.currentTime = time;
    setCurrentTime(time);
  };

  const togglePlay = () => {
    // スクロールのみモード中は先に停止
    if (scrollOnly) { stopScrollOnly(); }
    if (!audioRef.current || !session?.audioUrl) return;
    if (isPlaying) { audioRef.current.pause(); setIsPlaying(false); }
    else { audioRef.current.play(); setIsPlaying(true); }
  };

  // スクロールのみモード: 音なしでBPMベースで自動スクロール
  const toggleScrollOnly = () => {
    if (scrollOnly) {
      stopScrollOnly();
    } else {
      // 音声再生中なら停止
      if (isPlaying && audioRef.current) {
        audioRef.current.pause();
        setIsPlaying(false);
      }
      setScrollOnly(true);
      const startWall = performance.now();
      const startTime = currentTime;
      scrollStartRef.current = { startWall, startTime };
      const spd = speed;
      const tick = () => {
        const elapsed = (performance.now() - startWall) / 1000 * spd;
        setCurrentTime(startTime + elapsed);
        scrollTimerRef.current = requestAnimationFrame(tick);
      };
      scrollTimerRef.current = requestAnimationFrame(tick);
    }
  };

  const stopScrollOnly = () => {
    setScrollOnly(false);
    if (scrollTimerRef.current) {
      cancelAnimationFrame(scrollTimerRef.current);
      scrollTimerRef.current = null;
    }
  };

  const goHome = () => {
    stopScrollOnly();
    if (audioRef.current) { audioRef.current.pause(); audioRef.current.currentTime = 0; }
    setIsPlaying(false);
    setCurrentTime(0);
    setSession(null);
    setStatus(STATUS.IDLE);
    fetchHistory();
  };

  const formatTime = (s) => {
    if (isNaN(s) || !isFinite(s)) return "00:00";
    return new Date(s * 1000).toISOString().substr(14, 5);
  };

  const _showToast = (msg) => { setToast(msg); setTimeout(() => setToast(null), 3000); };

  const TUNING_GROUPS = [
    {
      label: "スタンダード系", options: [
        { value: "standard", label: "スタンダード (EADGBE)" },
        { value: "half_down", label: "半音下げ (E♭A♭D♭G♭B♭E♭)" },
        { value: "full_down", label: "全音下げ (DGCFAD)" },
      ]
    },
    {
      label: "Drop系", options: [
        { value: "drop_d", label: "Drop D (DADGBE)" },
        { value: "drop_c", label: "Drop C (CGCFAD)" },
        { value: "double_drop_d", label: "Double Drop D (DADGBD)" },
      ]
    },
    {
      label: "DADGAD系", options: [
        { value: "dadgad", label: "DADGAD" },
        { value: "dadgac", label: "DADGAC" },
        { value: "cgdgad", label: "CGDGAD" },
      ]
    },
    {
      label: "Open Major", options: [
        { value: "open_d", label: "Open D (DADF#AD)" },
        { value: "open_e", label: "Open E (EBEG#BE)" },
        { value: "open_g", label: "Open G (DGDGBD)" },
        { value: "open_a", label: "Open A (EAC#EAE)" },
        { value: "open_c", label: "Open C (CGCGCE)" },
      ]
    },
    {
      label: "Open Minor", options: [
        { value: "open_dm", label: "Open Dm (DADFAD)" },
        { value: "open_em", label: "Open Em (EBEGBE)" },
        { value: "open_gm", label: "Open Gm (DGDGBbD)" },
        { value: "open_am", label: "Open Am (EACEAE)" },
      ]
    },
    {
      label: "アーティスト系", options: [
        { value: "cgcgce", label: "CGCGCE (Sonic Youth)" },
        { value: "bebebe", label: "BEBEBE" },
        { value: "new_standard", label: "New Standard (CGDAEG)" },
      ]
    },
  ];
  const TUNING_OPTIONS = TUNING_GROUPS.flatMap(g => g.options);

  const handleRetune = async (newTuning, newCapo, newNoiseGate) => {
    if (!session?.id || retuning) return;
    const tuningToUse = newTuning || session.tuning || "standard";
    const capoToUse = newCapo !== undefined ? newCapo : capo;
    const gateToUse = newNoiseGate !== undefined ? newNoiseGate : noiseGate;
    setRetuning(true);
    try {
      const res = await fetch(`${API_BASE}/result/${session.id}/retune`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tuning: tuningToUse, capo: capoToUse, noise_gate: gateToUse }),
      });
      if (!res.ok) throw new Error("Retune failed");
      const data = await res.json();
      setSession(prev => ({ ...prev, tuning: tuningToUse, totalNotes: data.total_notes }));
      setRetuneKey(k => k + 1);
      if (newCapo !== undefined) {
        _showToast(`カポ ${capoToUse} に変更しました`);
      } else {
        _showToast(`チューニングを${TUNING_OPTIONS.find(t => t.value === tuningToUse)?.label || tuningToUse}に変更しました`);
      }
    } catch (err) {
      console.error('[handleRetune] failed:', err);
      _showToast("変更に失敗しました");
    } finally {
      setRetuning(false);
    }
  };

  const duration = audioRef.current?.duration || 0;
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  // Processing step definitions
  const STEPS = [
    { key: 'upload', label: 'アップロード完了', icon: '📁' },
    { key: 'beats', label: 'ビート検出', icon: '🥁' },
    { key: 'notes', label: 'ノート検出', icon: '🎵' },
    { key: 'strings', label: '弦・フレット推定', icon: '🎸' },
    { key: 'tab', label: 'TAB譜生成', icon: '📄' },
  ];

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--st-bg)', color: 'var(--st-text)', position: 'relative' }}
    >
      {session?.audioUrl && <audio ref={audioRef} src={session.audioUrl} preload="auto" crossOrigin="anonymous" onEnded={() => setIsPlaying(false)} />}
      <input ref={fileInputRef} type="file" accept=".mp3,.wav,.m4a,.flac" hidden
        onChange={(e) => { if (e.target.files?.[0]) handleUpload(e.target.files[0]); }} />

      {/* Full-screen D&D Overlay */}
      {isDragging && (
        <div className="drag-overlay">
          <UploadCloud size={120} className="drag-icon" />
          <h2>Drop to analyze</h2>
          <p>音声ファイルをドロップ</p>
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div className="app-logo" onClick={goHome}>
          <div className="logo-icon">
            <Music size={16} />
          </div>
          <span className="logo-text">SoloTab</span>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {status === STATUS.COMPLETED && (
            <button className="home-btn" onClick={goHome}>
              <Home size={14} style={{ marginRight: 4, verticalAlign: -2 }} />新規解析
            </button>
          )}
          <button onClick={toggleTheme} className="home-btn">
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
          </button>
        </div>
      </header>

      <div className="app-main">
        {/* ── IDLE: Landing Screen (NextChord style) ── */}
        {(status === STATUS.IDLE || status === STATUS.FAILED) && (
          <div className="upload-screen">
            <div className="ambient-glow" />

            {/* Hero Logo */}
            <div className="hero-logo-icon">
              <Music size={32} />
            </div>
            <h1 className="hero-title">SoloTab</h1>
            <p className="hero-subtitle">
              ソロギターをAIが瞬時にTAB譜へ。<br />
              <span className="sub-line">ノート検出・弦推定・TAB譜生成</span>
            </p>

            {/* Upload Card */}
            <div className="upload-card" onClick={() => fileInputRef.current?.click()}>
              <div className="icon-wrapper">
                <UploadCloud size={40} />
              </div>
              <h4>音源をドラッグ＆ドロップ</h4>
              <p>MP3, WAV, M4A, FLAC</p>
              <button className="select-btn" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                ファイルを選択
              </button>
            </div>

            {/* Solo Guitar Mode Toggle */}
            <label className="solo-toggle" onClick={(e) => e.stopPropagation()} style={{
              display: 'flex', alignItems: 'center', gap: 8, margin: '12px auto 0',
              cursor: 'pointer', fontSize: '0.85rem', color: 'var(--st-text-dim)',
              userSelect: 'none', width: 'fit-content',
            }}>
              <input type="checkbox" checked={soloGuitar} onChange={(e) => setSoloGuitar(e.target.checked)}
                style={{ accentColor: 'var(--st-accent)', width: 16, height: 16 }} />
              <span>🎸 ソロギターモード <span style={{ opacity: 0.6, fontSize: '0.75rem' }}>(Demucs分離スキップ・高速)</span></span>
            </label>


            {status === STATUS.FAILED && (
              <div className="error-message" style={{ marginTop: 16 }}>
                ❌ {progressMsg}
              </div>
            )}

            {history.length > 0 && (
              <div className="history-section">
                <h3><History size={12} /> 最近の解析</h3>
                {history.slice(0, 5).map(h => (
                  <div key={h.session_id} className="history-item" onClick={() => {
                    setSession({ id: h.session_id, fileName: h.filename });
                    restoreSession(h.session_id);
                  }}>
                    <div>
                      <div className="name">{h.filename}</div>
                      <div className="meta">
                        {h.total_notes ? `${h.total_notes} notes` : ''}{h.bpm ? ` · ${Math.round(h.bpm)} BPM` : ''}
                      </div>
                    </div>
                    <ChevronRight size={16} style={{ color: 'var(--st-text-dim)' }} />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── PROCESSING: Step-based Checklist (NextChord style) ── */}
        {(status === STATUS.UPLOADING || status === STATUS.PROCESSING) && (() => {
          const doneCount = stepsDone;
          const pct = Math.round((doneCount / STEPS.length) * 100);

          return (
            <div className="processing-screen">
              <div className="ambient-glow" />

              {/* Song info */}
              {session?.fileName && (
                <div className="processing-song-info">
                  <div className="name">{session.fileName}</div>
                </div>
              )}

              {/* Circular progress */}
              <div className="circular-progress">
                <svg width="100" height="100" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="42" fill="none" stroke="var(--st-surface-3)" strokeWidth="6" />
                  <circle cx="50" cy="50" r="42" fill="none" stroke="url(#stProgressGradient)" strokeWidth="6"
                    strokeLinecap="round" strokeDasharray={`${2 * Math.PI * 42}`}
                    strokeDashoffset={`${2 * Math.PI * 42 * (1 - pct / 100)}`}
                    transform="rotate(-90 50 50)"
                    style={{ transition: 'stroke-dashoffset 0.6s ease' }}
                  />
                  <defs>
                    <linearGradient id="stProgressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#f59e0b" />
                      <stop offset="100%" stopColor="#fb923c" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="pct">{pct}%</div>
              </div>

              {/* Step checklist */}
              <div className="step-checklist">
                {STEPS.map((step, i) => {
                  const isDone = i < doneCount;
                  const isCurrent = i === doneCount;
                  return (
                    <div key={step.key} className={`step-item ${isDone ? 'done' : isCurrent ? 'current' : 'pending'}`}>
                      <span className="step-icon">
                        {isDone ? '✅' : isCurrent ? step.icon : '⬜'}
                      </span>
                      <span className="step-label">{step.label}</span>
                      {isCurrent && <div className="step-spinner" />}
                      {isDone && <span className="step-done-tag">done</span>}
                    </div>
                  );
                })}
              </div>

              <p className="processing-footer">AI powered analysis</p>
            </div>
          );
        })()}

        {/* ── COMPLETED: Result ── */}
        {status === STATUS.COMPLETED && session && (
          <>
            {/* Compact Song Info Bar */}
            <div className="result-header" style={{
              display: 'flex', alignItems: 'center', gap: 12, padding: '8px 16px',
              background: 'var(--st-surface)', borderBottom: '1px solid var(--st-border)',
              flexWrap: 'wrap', minHeight: 44,
            }}>
              <h1 style={{ fontSize: 15, fontWeight: 700, margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 300 }}>
                {session.fileName || "Untitled"}
              </h1>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                {session.detectedKey && <span className="badge" style={{ color: '#10b981', fontSize: 11 }}>🎵 {session.detectedKey}</span>}
                {session.bpm && <span className="badge amber" style={{ fontSize: 11 }}>♩ {Math.round(session.bpm)}</span>}
                {session.totalNotes && <span className="badge accent" style={{ fontSize: 11 }}>♪ {session.totalNotes}</span>}
                {session.detectedCapo > 0 && <span className="badge" style={{ color: '#f59e0b', fontSize: 11 }}>Capo {session.detectedCapo}</span>}
              </div>
              <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginLeft: 'auto' }}>
                <select className="tuning-select" value={session.tuning || "standard"}
                  onChange={(e) => handleRetune(e.target.value)} disabled={retuning}
                  style={{ fontSize: 11, padding: '4px 6px', maxWidth: 160 }}>
                  {TUNING_GROUPS.map(group => (
                    <optgroup key={group.label} label={group.label}>
                      {group.options.map(t => (<option key={t.value} value={t.value}>{t.label}</option>))}
                    </optgroup>
                  ))}
                </select>
                <select className="tuning-select" value={capo}
                  onChange={(e) => { const v = Number(e.target.value); setCapo(v); handleRetune(null, v); }}
                  style={{ fontSize: 11, padding: '4px 6px', minWidth: 70 }}>
                  <option value={0}>カポなし</option>
                  {[1,2,3,4,5,6,7,8,9,10,11,12].map(n => (<option key={n} value={n}>Capo {n}</option>))}
                </select>
                <div className="transpose-controls" style={{ gap: 2 }}>
                  <button className="transpose-btn" onClick={() => setTranspose(t => t - 1)} style={{ width: 24, height: 24, fontSize: 14 }}>−</button>
                  <span className="transpose-label" style={{ fontSize: 11, minWidth: 28 }}>{transpose >= 0 ? '+' : ''}{transpose}</span>
                  <button className="transpose-btn" onClick={() => setTranspose(t => t + 1)} style={{ width: 24, height: 24, fontSize: 14 }}>+</button>
                </div>
                {retuning && <span style={{ fontSize: 10, color: 'var(--st-amber)' }}>⏳</span>}
                <button className="home-btn" title="PDF"
                  onClick={() => window.open(`${API_BASE}/result/${session.id}/pdf`, '_blank')}
                  style={{ fontSize: 11, padding: '4px 8px' }}>
                  <Printer size={12} style={{ marginRight: 2 }} />PDF
                </button>
                <button className="home-btn" title="Guitar Pro 5"
                  onClick={async () => {
                    try {
                      const res = await fetch(`${API_BASE}/result/${session.id}/gp5`);
                      if (!res.ok) throw new Error("取得失敗");
                      const blob = await res.blob();
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${(session.fileName || 'tab').replace(/\.[^.]+$/, '')}.gp5`;
                      a.style.display = 'none';
                      document.body.appendChild(a);
                      a.click();
                      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 200);
                    } catch(e) { _showToast("GP5: " + e.message); }
                  }}
                  style={{ fontSize: 11, padding: '4px 8px' }}>
                  <Download size={12} style={{ marginRight: 2 }} />GP5
                </button>
                <button className="home-btn" title="MusicXML"
                  onClick={async () => {
                    try {
                      const res = await fetch(`${API_BASE}/result/${session.id}/musicxml`);
                      if (!res.ok) throw new Error("取得失敗");
                      const blob = await res.blob();
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${(session.fileName || 'tab').replace(/\.[^.]+$/, '')}.musicxml`;
                      a.style.display = 'none';
                      document.body.appendChild(a);
                      a.click();
                      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 200);
                    } catch(e) { _showToast("MusicXML: " + e.message); }
                  }}
                  style={{ fontSize: 11, padding: '4px 8px' }}>
                  <Download size={12} style={{ marginRight: 2 }} />XML
                </button>
                <button className="home-btn" title="TuxGuitar用 GP4ダウンロード"
                  onClick={async () => {
                    try {
                      const res = await fetch(`${API_BASE}/result/${session.id}/gp4`);
                      if (!res.ok) throw new Error("取得失敗");
                      const blob = await res.blob();
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${(session.fileName || 'tab').replace(/\.[^.]+$/, '')}.gp4`;
                      a.style.display = 'none';
                      document.body.appendChild(a);
                      a.click();
                      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 200);
                      _showToast("TuxGuitar用GP4をダウンロードしました");
                    } catch(e) { _showToast("GP4: " + e.message); }
                  }}
                  style={{ fontSize: 11, padding: '4px 8px', color: '#10b981' }}>
                  🎸 TuxGuitar
                </button>
              </div>
            </div>

            {/* === Songsterr-Style Player Bar === */}
            <div className="player-control-bar" style={{
              display: 'flex', alignItems: 'center', background: '#252528', color: '#a0a0a5',
              padding: '0 16px', height: '64px', borderTop: '1px solid #1a1a1c', gap: '20px',
              fontFamily: 'Inter, sans-serif',
              position: 'fixed', bottom: 0, left: 0, width: '100%', zIndex: 100,
              boxSizing: 'border-box'
            }}>
              
              {/* 1. Play / Skip Controls */}
              <div style={{ display: 'flex', height: '40px', borderRadius: '4px', overflow: 'hidden', gap: '4px' }}>
                <button 
                  onClick={() => handleSeek(0)}
                  style={{ background: '#3a3a3c', border: 'none', width: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', borderRadius: '4px' }}
                  title="最初に戻る"
                >
                  <SkipBack size={20} color="#fff" />
                </button>
                <button 
                  onClick={togglePlay}
                  style={{ background: '#22c55e', border: 'none', width: '60px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', borderRadius: '4px' }}
                  title={isPlaying ? "一時停止" : "再生"}
                >
                  {isPlaying ? <Pause size={24} fill="currentColor" color="#000" /> : <Play size={24} fill="currentColor" color="#000" style={{ marginLeft: 3 }} />}
                </button>
                <button 
                  onClick={toggleScrollOnly}
                  style={{ background: scrollOnly ? '#f59e0b' : '#3a3a3c', border: 'none', width: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', borderRadius: '4px' }}
                  title={scrollOnly ? "スクロール停止" : "スクロールのみ（音なし）"}
                >
                  <span style={{ fontSize: '18px' }}>{scrollOnly ? '⏸' : '📜'}</span>
                </button>
              </div>

              {/* Central Timeline (Time + Draggable Progress) */}
              <div style={{ display: 'flex', alignItems: 'center', flexGrow: 1, gap: '12px', minWidth: '150px' }}>
                <span style={{ fontSize: '11px', fontWeight: 'bold', width: '36px', textAlign: 'right' }}>{formatTime(currentTime)}</span>
                
                <div style={{ position: 'relative', flexGrow: 1, height: '24px', display: 'flex', alignItems: 'center' }}>
                  {/* Visual Progress Bar (Background + Fill + Loop Markers) */}
                  <div style={{ position: 'absolute', width: '100%', height: '8px', background: '#1a1a1c', borderRadius: '4px', pointerEvents: 'none' }}>
                    <div style={{ width: `${progress}%`, background: '#22c55e', height: '100%', borderRadius: '4px' }} />
                    {loopA !== null && duration > 0 && <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(loopA/duration)*100}%`, width: '2px', background: '#4da6ff', zIndex: 10 }} />}
                    {loopB !== null && duration > 0 && <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(loopB/duration)*100}%`, width: '2px', background: '#4da6ff', zIndex: 10 }} />}
                    {loopA !== null && loopB !== null && duration > 0 && <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(loopA/duration)*100}%`, width: `${((loopB-loopA)/duration)*100}%`, background: 'rgba(77, 166, 255, 0.3)' }} />}
                  </div>
                  
                  {/* Interactive Range Slider overlay */}
                  <input 
                    type="range"
                    min={0}
                    max={duration || 100}
                    step={0.01}
                    value={currentTime}
                    onChange={(e) => handleSeek(Number(e.target.value))}
                    style={{
                      position: 'absolute',
                      width: '100%',
                      margin: 0,
                      opacity: 0, /* 透明にして見た目は下のバーを活かしつつ、操作はRangeで受ける */
                      cursor: 'pointer',
                      height: '100%'
                    }}
                  />
                </div>

                <span style={{ fontSize: '11px', fontWeight: 'bold', width: '36px' }}>{formatTime(duration)}</span>
              </div>

              {/* Right Toolbar Group */}
              <div style={{ display: 'flex', gap: '2px' }}>
                {/* 速度 (Speed) */}
                <button 
                  style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '48px', height: '48px', background: 'transparent', border: 'none', color: speed !== 1.0 ? '#4da6ff' : '#a0a0a5', cursor: 'pointer' }}
                  onClick={() => setSpeed(s => s === 1.0 ? 0.75 : s === 0.75 ? 0.5 : 1.0)}
                  title="速度 (クリックで変更)"
                >
                  <Activity size={18} />
                  <span style={{ fontSize: '9px', marginTop: '4px', fontWeight: 'bold' }}>{speed*100}%</span>
                </button>

                {/* ループ (Loop) */}
                <button 
                  style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '48px', height: '48px', background: 'transparent', border: 'none', color: (loopA !== null) ? '#22c55e' : '#a0a0a5', cursor: 'pointer' }}
                  onClick={() => {
                    if (loopA !== null && loopB !== null) { setLoopA(null); setLoopB(null); }
                    else if (loopA === null) setLoopA(currentTime);
                    else if (loopB === null && currentTime > loopA) setLoopB(currentTime);
                  }}
                  title={loopA === null ? "ループA点" : loopB === null ? "ループB点" : "ループ解除"}
                >
                  <Repeat size={18} />
                  <span style={{ fontSize: '9px', marginTop: '4px', fontWeight: 'bold' }}>ループ{loopA !== null && loopB === null ? ' (A)' : ''}</span>
                </button>

                {/* ======= NEW: NOISE CUT SLIDER ======= */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', margin: '0 10px', minWidth: '160px' }} title="AIのノイズ除去レベル。右にするほど細かい倍音ノイズが消えてシンプルになります">
                  <span style={{ fontSize: '9px', color: '#a0a0a5', fontWeight: 'bold', marginBottom: '2px' }}>CUT: {Math.round(noiseGate * 100)}%</span>
                  <input 
                    type="range" min="0" max="0.8" step="0.05" 
                    value={noiseGate} 
                    onChange={(e) => setNoiseGate(parseFloat(e.target.value))}
                    onMouseUp={(e) => handleRetune(null, null, parseFloat(e.target.value))}
                    onTouchEnd={(e) => handleRetune(null, null, parseFloat(e.target.value))}
                    style={{ width: '130px', accentColor: '#4da6ff', cursor: 'pointer' }}
                  />
                </div>
              </div>
            </div>

            {/* TAB View (Pushed up to allow space for the player bar without overlap) */}
            <div className="tab-container" style={{ paddingBottom: '64px' }}>
              <TabView
                key={retuneKey}
                sessionId={session.id}
                apiBase={API_BASE}
                currentTime={currentTime}
                isPlaying={isPlaying || scrollOnly}
                transpose={transpose}
                capo={capo}
              />
            </div>
          </>
        )}
      </div>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

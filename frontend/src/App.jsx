import React, { useState, useEffect, useRef } from "react";
import { UploadCloud, Play, Pause, Square, History, Music, ChevronRight, ChevronLeft, Sun, Moon, Home, Download, Printer, Activity, Repeat, Headphones, VolumeX, Timer, MoreVertical, Edit2, ChevronUp, SkipBack, ArrowLeft } from "lucide-react";
import { TabView } from "./components/TabView";

// ── API接続先の動的解決 ──
// ローカル → 環境変数 or localhost:8002
// 外部トンネル → tunnel_urls.json から solotab_be を取得
const _envUrl = import.meta.env.VITE_API_URL;
let API_BASE = _envUrl !== undefined ? _envUrl : "http://localhost:8002";
let PORTAL_URL = "https://guitar-suite.vercel.app/portal";

// tunnel_urls.json から接続先URLを非同期取得
const _tunnelApiPromise = fetch('/tunnel_urls.json')
  .then(r => r.json())
  .then(data => {
    if (data?.solotab_be) { API_BASE = data.solotab_be; console.log('[SoloTab] Using tunnel API:', API_BASE); }
    if (data?.portal) { PORTAL_URL = data.portal; console.log('[SoloTab] Using portal URL:', PORTAL_URL); }
  })
  .catch(() => console.log('[SoloTab] tunnel_urls.json not found, using default API'));

const STATUS = { IDLE: "idle", UPLOADING: "uploading", PROCESSING: "processing", COMPLETED: "completed", FAILED: "failed" };

export default function SoloTabApp() {
  const [status, setStatus] = useState(STATUS.IDLE);
  const [progressMsg, setProgressMsg] = useState("");
  const [stepsDone, setStepsDone] = useState(0);
  const [session, setSession] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [noiseGate, setNoiseGate] = useState(0.10);
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
  const [guitarType, setGuitarType] = useState("auto");
  const [transProfile, setTransProfile] = useState(() => {
    try { return localStorage.getItem('solotab-profile') || 'standard'; } catch { return 'standard'; }
  });
  const [techGp5, setTechGp5] = useState(() => {
    try { return localStorage.getItem('solotab-tech-gp5') === 'true'; } catch { return false; }
  });
  const [techOverlay, setTechOverlay] = useState(() => {
    try { return localStorage.getItem('solotab-tech-overlay') === 'true'; } catch { return false; }
  });
  const [techFingers, setTechFingers] = useState(() => {
    try { return localStorage.getItem('solotab-tech-fingers') === 'true'; } catch { return false; }
  });
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('solotab-theme') || 'dark'; } catch { return 'dark'; }
  });

  const [metronomeEnabled, setMetronomeEnabled] = useState(false);
  const [syncOffset, setSyncOffset] = useState(0);
  const [tempoMultiplier, setTempoMultiplier] = useState(1.0);

  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    let timer;
    if (status === STATUS.PROCESSING || status === STATUS.UPLOADING) {
      setElapsedSeconds(0);
      timer = setInterval(() => {
        setElapsedSeconds(prev => prev + 1);
      }, 1000);
    } else {
      setElapsedSeconds(0);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [status]);

  const audioRef = useRef(null);
  const sseRef = useRef(null);
  const fileInputRef = useRef(null);
  const scrollTimerRef = useRef(null);
  const scrollStartRef = useRef(null);
  const [scrollOnly, setScrollOnly] = useState(false);

  const [processingAudioPlaying, setProcessingAudioPlaying] = useState(false);
  const [serverOnline, setServerOnline] = useState(true);
  const uploadLockRef = useRef(false);
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

  // サーバー接続確認（処理中はスキップ：SSE/ポーリングで接続確認済み）
  useEffect(() => {
    const checkServer = async () => {
      // 処理中・アップロード中はGPU負荷でタイムアウトしやすいのでスキップ
      if (status === STATUS.PROCESSING || status === STATUS.UPLOADING) {
        setServerOnline(true); // SSEが動いている=接続OK
        return;
      }
      try {
        const res = await fetch(`${API_BASE}/sessions`, { signal: AbortSignal.timeout(15000) });
        setServerOnline(res.ok);
      } catch {
        setServerOnline(false);
      }
    };
    checkServer();
    const iv = setInterval(checkServer, 60000);
    return () => clearInterval(iv);
  }, [status]);

  // 処理中にブラウザを閉じる/リロードする場合の警告
  useEffect(() => {
    const handler = (e) => {
      if (status === STATUS.PROCESSING || status === STATUS.UPLOADING) {
        e.preventDefault();
        e.returnValue = '処理中です。ページを離れますか？';
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [status]);

  // 起動時: 常にIDLE状態で開始し、履歴を取得
  // ※ ポータルからの画面遷移時に自動解析が走らないようにする
  useEffect(() => {
    // 古い処理中セッションのキーをクリーンアップ
    localStorage.removeItem('solotab-processing-session');
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
      // カポは自動適用しない: ユーザーが手動で選択する
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
        if (typeof data.steps_done === 'number') setStepsDone(prev => Math.max(prev, data.steps_done));
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
      setProgressMsg("接続が切れました。再接続中...");
      // fallback polling with retry limit
      let retries = 0;
      const MAX_RETRIES = 60; // 2s × 60 = 2分間リトライ
      const poll = setInterval(async () => {
        retries++;
        if (retries > MAX_RETRIES) {
          clearInterval(poll);
          // 最終確認: 完了しているかも
          try {
            const res = await fetch(`${API_BASE}/status/${sid}`);
            const data = await res.json();
            if (data.status === "completed") { handleCompleted(sid); return; }
          } catch {}
          setStatus(STATUS.FAILED);
          setProgressMsg("サーバーとの接続がタイムアウトしました。ページをリロードしてください。");
          localStorage.removeItem('solotab-processing-session');
          return;
        }
        try {
          const res = await fetch(`${API_BASE}/status/${sid}`);
          const data = await res.json();
          setProgressMsg(data.progress || "解析中...");
          if (typeof data.steps_done === 'number') setStepsDone(prev => Math.max(prev, data.steps_done));
          if (data.status === "completed") { clearInterval(poll); handleCompleted(sid); }
          else if (data.status === "failed") {
            clearInterval(poll);
            setStatus(STATUS.FAILED);
            setProgressMsg(data.error || "解析に失敗しました");
            localStorage.removeItem('solotab-processing-session');
          }
        } catch { /* polling error, retry next interval */ }
      }, 2000);
    };
  };

  const handleCompleted = async (sid) => {
    localStorage.removeItem('solotab-processing-session');
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
      setStatus(STATUS.COMPLETED);
    } catch {
      setStatus(STATUS.FAILED);
      setProgressMsg("結果取得に失敗");
    }
  };

  // Upload
  const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200MB
  const handleUpload = async (file) => {
    if (!file) return;
    // 二重送信防止
    if (uploadLockRef.current) return;
    // サーバー接続チェック
    if (!serverOnline) {
      setStatus(STATUS.FAILED);
      setProgressMsg('サーバーに接続できません。quick_start.bat でサーバーを起動してください。');
      return;
    }
    // ファイルサイズチェック
    if (file.size > MAX_FILE_SIZE) {
      setStatus(STATUS.FAILED);
      setProgressMsg(`ファイルが大きすぎます (${(file.size / 1024 / 1024).toFixed(0)}MB)。200MB以下のファイルを使用してください。`);
      return;
    }
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
    // 前の処理が走っていたらSSEを切断
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
    uploadLockRef.current = true;
    setStatus(STATUS.UPLOADING);
    setProgressMsg("アップロード中...");
    setStepsDone(0);
    setIsPlaying(false);
    setProcessingAudioPlaying(false);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("skip_demucs", soloGuitar);
    formData.append("fast_moe", "true");
    formData.append("guitar_type", guitarType);
    formData.append("transcription_profile", transProfile);
    formData.append("enable_technique_gp5", techGp5);
    formData.append("enable_technique_overlay", techOverlay);
    formData.append("enable_technique_fingers", techFingers);
    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      setSession({ id: data.session_id, fileName: file.name, audioUrl: `${API_BASE}${data.audio_url}` });
      setStatus(STATUS.PROCESSING);
      setStepsDone(1); // アップロード完了 = 20%
      setProgressMsg("ビート検出中...");
      localStorage.setItem('solotab-processing-session', data.session_id);
      startStatusStream(data.session_id);
    } catch (err) {
      setStatus(STATUS.FAILED);
      setProgressMsg(err.message || "アップロードに失敗しました");
    } finally {
      uploadLockRef.current = false;
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
        body: JSON.stringify({ url: urlToUse.trim(), guitar_type: guitarType, transcription_profile: transProfile }),
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
    // 処理中に新しいファイルをドロップ → 確認
    if (status === STATUS.PROCESSING || status === STATUS.UPLOADING) {
      if (!confirm('現在解析中です。新しいファイルでやり直しますか？')) return;
    }
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
    // 処理中にホームに戻る → 確認
    if (status === STATUS.PROCESSING || status === STATUS.UPLOADING) {
      if (!confirm('解析中です。ホームに戻りますか？\n(バックグラウンドで処理は続行します)')) return;
    }
    stopScrollOnly();
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
    if (audioRef.current) { audioRef.current.pause(); audioRef.current.currentTime = 0; }
    setIsPlaying(false);
    setProcessingAudioPlaying(false);
    setCurrentTime(0);
    setSession(null);
    setStatus(STATUS.IDLE);
    setStepsDone(0);
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
    // Always fallback to current session values — never send null/undefined to backend
    const tuningToUse = newTuning || session.tuning || "standard";
    const capoToUse = (newCapo !== undefined && newCapo !== null) ? newCapo : capo;
    const gateToUse = (newNoiseGate !== undefined && newNoiseGate !== null) ? newNoiseGate : noiseGate;
    setRetuning(true);
    try {
      const res = await fetch(`${API_BASE}/result/${session.id}/retune`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tuning: tuningToUse, capo: capoToUse, noise_gate: gateToUse }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Retune failed (${res.status})`);
      }
      const data = await res.json();
      console.log('[handleRetune] Backend response:', data);
      // Sync all state with backend — capo, noiseGate, tuning, totalNotes, badge
      setCapo(capoToUse);
      setNoiseGate(gateToUse);
      setSession(prev => ({
        ...prev,
        tuning: tuningToUse,
        totalNotes: data.total_notes,
        detectedCapo: capoToUse,  // update badge display
      }));
      // Wait for backend to flush GP5 file to disk before re-fetching
      await new Promise(r => setTimeout(r, 500));
      setRetuneKey(k => k + 1);
      if (newCapo !== undefined && newCapo !== null) {
        _showToast(capoToUse > 0 ? `カポ ${capoToUse} に変更しました` : 'カポを外しました');
      } else if (newNoiseGate !== undefined && newNoiseGate !== null) {
        _showToast(`CUT: ${Math.round(gateToUse * 100)}% に変更`);
      } else {
        _showToast(`チューニングを${TUNING_OPTIONS.find(t => t.value === tuningToUse)?.label || tuningToUse}に変更しました`);
      }
    } catch (err) {
      console.error('[handleRetune] failed:', err);
      _showToast(`変更に失敗: ${err.message}`);
    } finally {
      setRetuning(false);
    }
  };

  const duration = audioRef.current?.duration || 0;
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  const originalBpm = session?.bpm || 120;
  const currentBpm = Math.round(originalBpm * speed);

  const handleBpmChange = (newBpm) => {
    if (!originalBpm) return;
    const minBpm = Math.round(originalBpm * 0.5);
    const maxBpm = Math.round(originalBpm * 2.0);
    const clampedBpm = Math.max(minBpm, Math.min(maxBpm, newBpm));
    const newRate = clampedBpm / originalBpm;
    setSpeed(Math.max(0.5, Math.min(2.0, Math.round(newRate * 100) / 100)));
  };

  const decreaseBpm = () => handleBpmChange(currentBpm - 1);
  const increaseBpm = () => handleBpmChange(currentBpm + 1);
  const resetBpm = () => setSpeed(1.0);

  // Processing step definitions
  const STEPS = [
    { key: 'upload', label: 'アップロード完了', icon: '📁' },
    { key: 'beats', label: 'ビート検出', icon: '🥁' },
    { key: 'notes', label: 'ノート検出', icon: '🎵' },
    { key: 'strings', label: '弦・フレット推定', icon: '🎸' },
    { key: 'tab', label: 'TAB譜生成', icon: '📄' },
  ];

  // Reset audio when entering processing
  useEffect(() => {
    if (status === STATUS.PROCESSING || status === STATUS.UPLOADING) {
      setProcessingAudioPlaying(false);
    }
  }, [status]);

  const toggleProcessingAudio = () => {
    if (!audioRef.current || !session?.audioUrl) return;
    if (processingAudioPlaying) {
      audioRef.current.pause();
      setProcessingAudioPlaying(false);
    } else {
      audioRef.current.play();
      setProcessingAudioPlaying(true);
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--st-bg)', color: 'var(--st-text)', position: 'relative' }}
    >
      {/* Processing screen keyframes */}
      <style>{`
        @keyframes processingPulse1 {
          0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.15; }
          50% { transform: translate(-50%, -50%) scale(1.3); opacity: 0.05; }
        }
        @keyframes processingPulse2 {
          0%, 100% { transform: translate(-50%, -50%) scale(1.1); opacity: 0.10; }
          50% { transform: translate(-50%, -50%) scale(1.5); opacity: 0.03; }
        }
        @keyframes processingPulse3 {
          0%, 100% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.07; }
          50% { transform: translate(-50%, -50%) scale(1.7); opacity: 0.01; }
        }
        @keyframes waveBar {
          0%, 100% { transform: scaleY(0.3); }
          50% { transform: scaleY(1); }
        }
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        .proc-tip-enter {
          opacity: 1;
          transform: translateY(0);
          transition: opacity 0.4s ease, transform 0.4s ease;
        }
        .proc-tip-exit {
          opacity: 0;
          transform: translateY(8px);
          transition: opacity 0.3s ease, transform 0.3s ease;
        }
        .proc-mini-play:hover {
          transform: scale(1.1) !important;
          box-shadow: 0 0 20px rgba(245, 158, 11, 0.4) !important;
        }
        .proc-mini-play:active {
          transform: scale(0.95) !important;
        }
      `}</style>
      {session?.audioUrl && <audio ref={audioRef} src={session.audioUrl} preload="auto" crossOrigin="anonymous" onEnded={() => { setIsPlaying(false); setProcessingAudioPlaying(false); }} />}
      <input ref={fileInputRef} type="file" accept=".mp3,.wav,.m4a,.flac" hidden
        onChange={(e) => { if (e.target.files?.[0]) handleUpload(e.target.files[0]); }} />

      {isDragging && (
        <div className="drag-overlay">
          <UploadCloud size={120} className="drag-icon" />
          <h2>Drop to analyze</h2>
          <p>音声ファイルをドロップ</p>
        </div>
      )}

      {/* Server offline banner */}
      {!serverOnline && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999,
          background: 'linear-gradient(135deg, #dc2626, #b91c1c)',
          color: 'white', padding: '10px 20px',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
          fontSize: 14, fontWeight: 600,
          boxShadow: '0 2px 12px rgba(220,38,38,0.4)',
        }}>
          <span style={{ fontSize: 18 }}>⚠️</span>
          サーバーに接続できません。quick_start.bat でサーバーを起動してください。
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <a
            href={PORTAL_URL}
            className="portal-back-btn"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '12px',
              fontWeight: 600,
              padding: '6px 12px',
              borderRadius: '20px',
              background: 'rgba(255,255,255,0.10)',
              border: '1px solid rgba(255,255,255,0.10)',
              color: 'var(--st-text)',
              textDecoration: 'none',
              transition: 'all 0.2s',
            }}
            title="ポータルに戻る"
          >
            <ArrowLeft size={14} />
            <span>ポータル</span>
          </a>
          <div className="app-logo" onClick={goHome} role="button" tabIndex={0} aria-label="Go to home screen" onKeyDown={(e) => { if (e.key === 'Enter') goHome(); }}>
            <div className="logo-icon">
              <Music size={16} style={{ color: 'white' }} />
            </div>
            <span className="logo-text">SoloTab</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          {status === STATUS.COMPLETED && (
            <button className="home-btn" onClick={goHome}>
              <Home size={14} style={{ marginRight: 4, verticalAlign: -2 }} />新規解析
            </button>
          )}
          <button
            onClick={toggleTheme}
            style={{
              width: 36,
              height: 36,
              borderRadius: 10,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'transparent',
              border: '1px solid var(--st-border)',
              color: 'var(--st-text-dim)',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            title={theme === 'dark' ? 'ライトモードに切替' : 'ダークモードに切替'}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--st-surface-2)'; e.currentTarget.style.color = theme === 'dark' ? '#fbbf24' : '#6366f1'; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--st-text-dim)'; }}
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
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
              <Music size={32} style={{ color: 'white' }} />
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

            {/* Options Section */}
            <div style={{ width: '100%', maxWidth: 448, marginTop: 24 }}>
              {/* Divider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, color: 'var(--st-text-muted)' }}>
                <div style={{ flex: 1, height: 1, background: 'var(--st-border)' }} />
                <span style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.2em' }}>オプション</span>
                <div style={{ flex: 1, height: 1, background: 'var(--st-border)' }} />
              </div>

              {/* Solo Guitar Mode Toggle */}
              <label onClick={(e) => e.stopPropagation()} style={{
                display: 'flex', alignItems: 'center', gap: 10, margin: '0 0 10px',
                cursor: 'pointer', fontSize: '0.85rem', color: 'var(--st-text-dim)',
                userSelect: 'none',
                padding: '10px 16px', borderRadius: 12,
                background: 'var(--st-surface)', border: '1px solid var(--st-border)',
                transition: 'all 0.2s',
              }}>
                <input type="checkbox" checked={soloGuitar} onChange={(e) => setSoloGuitar(e.target.checked)}
                  style={{ accentColor: 'var(--st-accent)', width: 16, height: 16, flexShrink: 0 }} />
                <span>🎸 ソロギターモード <span style={{ opacity: 0.6, fontSize: '0.75rem' }}>(Demucs分離スキップ・高速)</span></span>
              </label>

              {/* Guitar Type Selector */}
              <div onClick={(e) => e.stopPropagation()} style={{
                display: 'flex', alignItems: 'center', gap: 10, margin: '0 0 10px',
                fontSize: '0.85rem', color: 'var(--st-text-dim)',
                userSelect: 'none',
                padding: '10px 16px', borderRadius: 12,
                background: 'var(--st-surface)', border: '1px solid var(--st-border)',
              }}>
                <span>🎵 弦タイプ:</span>
                <select value={guitarType} onChange={(e) => setGuitarType(e.target.value)}
                  style={{ fontSize: '0.85rem', padding: '4px 8px', borderRadius: 8,
                    background: 'var(--st-surface-2)', color: 'var(--st-text)',
                    border: '1px solid var(--st-border)', cursor: 'pointer' }}>
                  <option value="auto">自動判別</option>
                  <option value="steel">アコギ (スチール弦)</option>
                  <option value="nylon">クラシック (ナイロン弦)</option>
                </select>
              </div>

              {/* Transcription Profile Selector */}
              <div onClick={(e) => e.stopPropagation()} style={{
                display: 'flex', alignItems: 'center', gap: 10, margin: '0 0 10px',
                fontSize: '0.85rem', color: 'var(--st-text-dim)',
                userSelect: 'none',
                padding: '10px 16px', borderRadius: 12,
                background: 'var(--st-surface)', border: '1px solid var(--st-border)',
              }}>
                <span>🎼 解析モード:</span>
                <select value={transProfile} onChange={(e) => {
                  setTransProfile(e.target.value);
                  localStorage.setItem('solotab-profile', e.target.value);
                }}
                  style={{ fontSize: '0.85rem', padding: '4px 8px', borderRadius: 8,
                    background: 'var(--st-surface-2)', color: 'var(--st-text)',
                    border: '1px solid var(--st-border)', cursor: 'pointer' }}>
                  <option value="standard">標準ソロギター (ストローク/ソロ)</option>
                  <option value="classic">クラシック・アルペジオ (繊細・分散和音)</option>
                </select>
              </div>
              <div onClick={(e) => e.stopPropagation()} style={{
                display: 'flex', flexDirection: 'column', gap: 6, margin: '0',
                fontSize: '0.8rem', color: 'var(--st-text-dim)',
                userSelect: 'none',
                padding: '12px 16px', borderRadius: 12,
                background: 'var(--st-surface)', border: '1px solid var(--st-border)',
              }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--st-accent)', marginBottom: 4 }}>🧪 テクニック表示（実験的）</span>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', padding: '2px 0' }}>
                  <input type="checkbox" checked={techGp5}
                    onChange={(e) => { setTechGp5(e.target.checked); localStorage.setItem('solotab-tech-gp5', e.target.checked); }}
                    style={{ accentColor: 'var(--st-accent)', width: 14, height: 14, flexShrink: 0 }} />
                  TABに記号書込 (H/P/S/C)
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', padding: '2px 0' }}>
                  <input type="checkbox" checked={techOverlay}
                    onChange={(e) => { setTechOverlay(e.target.checked); localStorage.setItem('solotab-tech-overlay', e.target.checked); }}
                    style={{ accentColor: 'var(--st-accent)', width: 14, height: 14, flexShrink: 0 }} />
                  オーバーレイ表示
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', padding: '2px 0' }}>
                  <input type="checkbox" checked={techFingers}
                    onChange={(e) => { setTechFingers(e.target.checked); localStorage.setItem('solotab-tech-fingers', e.target.checked); }}
                    style={{ accentColor: 'var(--st-accent)', width: 14, height: 14, flexShrink: 0 }} />
                  テクニック連動指修正
                </label>
              </div>
            </div>

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
                    if (status === STATUS.PROCESSING || status === STATUS.UPLOADING) {
                      if (!confirm('解析中です。別の曲に切り替えますか？')) return;
                      if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
                    }
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

        {/* ── PROCESSING: Enhanced Step-based Checklist ── */}
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

              {/* Circular progress with wave animation */}
              <div style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 24 }}>
                {/* Pulsing wave rings */}
                <div style={{
                  position: 'absolute', width: 140, height: 140, borderRadius: '50%',
                  border: '2px solid rgba(245, 158, 11, 0.15)',
                  animation: 'processingPulse1 3s ease-in-out infinite',
                  top: '50%', left: '50%',
                }} />
                <div style={{
                  position: 'absolute', width: 170, height: 170, borderRadius: '50%',
                  border: '1.5px solid rgba(251, 146, 60, 0.10)',
                  animation: 'processingPulse2 3.5s ease-in-out infinite',
                  top: '50%', left: '50%',
                }} />
                <div style={{
                  position: 'absolute', width: 200, height: 200, borderRadius: '50%',
                  border: '1px solid rgba(245, 158, 11, 0.06)',
                  animation: 'processingPulse3 4s ease-in-out infinite',
                  top: '50%', left: '50%',
                }} />

                <div className="circular-progress" style={{ position: 'relative', zIndex: 2, marginBottom: 0 }}>
                  <svg width="120" height="120" viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="50" fill="none" stroke="var(--st-surface-3)" strokeWidth="5" />
                    <circle cx="60" cy="60" r="50" fill="none" stroke="url(#stProgressGradient)" strokeWidth="5"
                      strokeLinecap="round" strokeDasharray={`${2 * Math.PI * 50}`}
                      strokeDashoffset={`${2 * Math.PI * 50 * (1 - pct / 100)}`}
                      transform="rotate(-90 60 60)"
                      style={{ transition: 'stroke-dashoffset 0.6s ease', filter: 'drop-shadow(0 0 6px rgba(245, 158, 11, 0.3))' }}
                    />
                    <defs>
                      <linearGradient id="stProgressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#f59e0b" />
                        <stop offset="100%" stopColor="#fb923c" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="pct" style={{ fontSize: 24 }}>{pct}%</div>
                </div>
              </div>

              {/* Detailed status message */}
              <div style={{
                fontSize: 15, fontWeight: 700, color: 'var(--st-text)',
                marginBottom: 16, textAlign: 'center', minHeight: 24,
                background: 'linear-gradient(90deg, #f59e0b, #fb923c)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>
                {progressMsg || "解析中..."}
              </div>

              {/* Step progress message */}
              <div style={{
                fontSize: 13, fontWeight: 600, color: 'var(--st-accent)',
                marginBottom: 8, letterSpacing: '0.02em',
                display: 'flex', alignItems: 'center', gap: 12,
              }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span>⏱</span>
                  ステップ {doneCount}/{STEPS.length}
                </span>
                <span style={{ color: 'var(--st-text-dim)', fontWeight: 500 }}>
                  経過時間: {Math.floor(elapsedSeconds / 60)}分{(elapsedSeconds % 60).toString().padStart(2, '0')}秒
                </span>
              </div>

              {/* Mini audio player */}
              {session?.audioUrl && (
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 20px', borderRadius: 16,
                  background: 'var(--st-surface)', border: '1px solid var(--st-border)',
                  marginBottom: 16, minWidth: 200,
                }}>
                  <button
                    className="proc-mini-play"
                    onClick={toggleProcessingAudio}
                    style={{
                      width: 36, height: 36, borderRadius: '50%',
                      background: 'var(--st-gradient-brand)',
                      border: 'none', cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: 'white', flexShrink: 0,
                      transition: 'all 0.2s',
                      boxShadow: '0 2px 10px rgba(245, 158, 11, 0.25)',
                    }}
                    title={processingAudioPlaying ? '一時停止' : '待ち時間に曲を聴く'}
                  >
                    {processingAudioPlaying ? <Pause size={16} /> : <Play size={16} style={{ marginLeft: 2 }} />}
                  </button>
                  {/* Waveform bars animation */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 2, height: 24 }}>
                    {[0, 1, 2, 3, 4, 5, 6, 7, 8].map(i => (
                      <div key={i} style={{
                        width: 3, borderRadius: 2,
                        height: processingAudioPlaying ? '100%' : 8,
                        background: processingAudioPlaying
                          ? `linear-gradient(to top, #f59e0b, #fb923c)`
                          : 'var(--st-surface-3)',
                        animation: processingAudioPlaying
                          ? `waveBar ${0.4 + (i % 3) * 0.15}s ease-in-out ${i * 0.05}s infinite alternate`
                          : 'none',
                        transition: 'height 0.3s ease, background 0.3s ease',
                        transformOrigin: 'bottom',
                      }} />
                    ))}
                  </div>
                  <span style={{
                    fontSize: 11, color: 'var(--st-text-dim)', fontWeight: 500,
                    whiteSpace: 'nowrap',
                  }}>
                    {processingAudioPlaying ? '再生中 🎧' : '曲を聴く'}
                  </span>
                </div>
              )}

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



            </div>
          );
        })()}

        {/* ── COMPLETED: Result ── */}
        {status === STATUS.COMPLETED && session && (
          <>
            {/* 1. Song Header (NextChord Style) */}
            <div className="result-header" style={{
              display: 'flex', alignItems: 'center', gap: 12, padding: '12px 24px',
              background: 'var(--st-surface)', borderBottom: '1px solid var(--st-border)',
              flexWrap: 'wrap', minHeight: 44,
            }}>
              <h1 style={{ fontSize: 16, fontWeight: 800, margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 400 }}>
                {session.fileName || "Untitled Track"}
              </h1>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginLeft: 8 }}>
                {session.detectedKey && <span className="badge" style={{ color: '#10b981', fontSize: 11 }}>Key: {session.detectedKey}</span>}
                {session.bpm && <span className="badge amber" style={{ fontSize: 11 }}>♩ {Math.round(session.bpm)} BPM</span>}
                {session.totalNotes && <span className="badge accent" style={{ fontSize: 11 }}>♪ {session.totalNotes} notes</span>}
                {session.detectedCapo > 0 && <span className="badge" style={{
                  color: '#fff', fontSize: 11, fontWeight: 700,
                  background: 'linear-gradient(135deg, #f59e0b, #d97706)',
                  padding: '2px 8px', borderRadius: 10,
                }}>🎸 Capo {session.detectedCapo} (AI推定)</span>}
              </div>
            </div>

            {/* 2. Playback Control Bar (NextChord Style) */}
            <div className="player-control-bar" style={{
              display: 'flex', alignItems: 'center', background: '#252528', color: '#a0a0a5',
              padding: '0 24px', height: '64px', borderBottom: '1px solid #1a1a1c', gap: '20px',
              fontFamily: 'Inter, sans-serif',
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

              {/* Central Timeline */}
              <div style={{ display: 'flex', alignItems: 'center', flexGrow: 1, gap: '12px' }}>
                <span style={{ fontSize: '11px', fontWeight: 'bold', width: '36px', textAlign: 'right' }}>{formatTime(currentTime)}</span>
                
                <div style={{ position: 'relative', flexGrow: 1, height: '24px', display: 'flex', alignItems: 'center' }}>
                  <div style={{ position: 'absolute', width: '100%', height: '8px', background: '#1a1a1c', borderRadius: '4px', pointerEvents: 'none' }}>
                    <div style={{ width: `${progress}%`, background: '#22c55e', height: '100%', borderRadius: '4px' }} />
                    {loopA !== null && duration > 0 && <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(loopA/duration)*100}%`, width: '2px', background: '#4da6ff', zIndex: 10 }} />}
                    {loopB !== null && duration > 0 && <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(loopB/duration)*100}%`, width: '2px', background: '#4da6ff', zIndex: 10 }} />}
                    {loopA !== null && loopB !== null && duration > 0 && <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${(loopA/duration)*100}%`, width: `${((loopB-loopA)/duration)*100}%`, background: 'rgba(77, 166, 255, 0.3)' }} />}
                  </div>
                  
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
                      opacity: 0,
                      cursor: 'pointer',
                      height: '100%'
                    }}
                  />
                </div>

                <span style={{ fontSize: '11px', fontWeight: 'bold', width: '36px' }}>{formatTime(duration)}</span>
              </div>
            </div>

            {/* 3. Control Ribbon (NextChord Style, SoloTab Features) */}
            <div className="control-ribbon scrollbar-hide">
              {/* ① Speed (BPM Linked) */}
              <div className="ribbon-item">
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
                  <button onClick={decreaseBpm} title="BPMを下げる" style={{ background: 'transparent', border: 'none', color: 'var(--st-text)', cursor: 'pointer', display: 'flex', alignItems: 'center' }}><ChevronLeft size={14} /></button>
                  <span 
                    style={{ fontSize: 13, fontWeight: 900, fontStyle: 'italic', color: 'var(--st-text)', width: 60, textAlign: 'center', cursor: 'pointer' }}
                    onDoubleClick={resetBpm}
                    title="ダブルクリックで元のテンポにリセット"
                  >
                    ♩ {currentBpm}
                  </span>
                  <button onClick={increaseBpm} title="BPMを上げる" style={{ background: 'transparent', border: 'none', color: 'var(--st-text)', cursor: 'pointer', display: 'flex', alignItems: 'center' }}><ChevronRight size={14} /></button>
                </div>
                <span style={{ fontSize: 9, color: 'var(--st-text-dim)', fontWeight: 'bold' }}>速度: {Math.round(speed * 100)}%</span>
              </div>

              <div className="ribbon-divider" />

              {/* ② Loop */}
              <button 
                onClick={() => {
                  if (loopA !== null && loopB !== null) { setLoopA(null); setLoopB(null); }
                  else if (loopA === null) setLoopA(currentTime);
                  else if (loopB === null && currentTime > loopA) setLoopB(currentTime);
                }}
                className={`ribbon-btn ${loopA !== null ? 'active' : ''}`}
                title={loopA === null ? "ループA点" : loopB === null ? "ループB点" : "ループ解除"}
              >
                <Repeat size={18} />
                <span style={{ fontSize: 9, marginTop: 4, fontWeight: 'bold' }}>ループ{loopA !== null && loopB === null ? ' (A)' : ''}</span>
              </button>

              <div className="ribbon-divider" />

              {/* ③ Tuning */}
              <div className="ribbon-item">
                <select className="tuning-select" value={session.tuning || "standard"}
                  onChange={(e) => handleRetune(e.target.value)} disabled={retuning}
                  style={{ fontSize: 11, padding: '4px 6px', maxWidth: 160, marginBottom: 2 }}>
                  {TUNING_GROUPS.map(group => (
                    <optgroup key={group.label} label={group.label}>
                      {group.options.map(t => (<option key={t.value} value={t.value}>{t.label}</option>))}
                    </optgroup>
                  ))}
                </select>
                <span style={{ fontSize: 9, color: 'var(--st-text-dim)', fontWeight: 'bold' }}>チューニング {retuning && <span style={{ color: 'var(--st-amber)' }}>⏳</span>}</span>
              </div>

              <div className="ribbon-divider" />

              {/* ④ Capo */}
              <div className="ribbon-item">
                <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', marginBottom: 2 }}>
                  <select className="tuning-select" value={capo}
                    onChange={(e) => { const v = Number(e.target.value); setCapo(v); handleRetune(null, v); }}
                    style={{
                      fontSize: 11, padding: '4px 6px', minWidth: 80,
                      border: capo > 0 ? '1.5px solid #f59e0b' : undefined,
                      boxShadow: capo > 0 ? '0 0 6px rgba(245,158,11,0.3)' : undefined,
                    }}>
                    <option value={0}>カポなし</option>
                    {[1,2,3,4,5,6,7,8,9,10,11,12].map(n => (
                      <option key={n} value={n}>
                        Capo {n}{session.detectedCapo === n ? ' ★AI' : ''}
                      </option>
                    ))}
                  </select>
                  {session.detectedCapo > 0 && capo !== session.detectedCapo && (
                    <button
                      onClick={() => { setCapo(session.detectedCapo); handleRetune(null, session.detectedCapo); }}
                      title={`AI推定: Capo ${session.detectedCapo} に戻す`}
                      style={{
                        marginLeft: 4, padding: '2px 6px', borderRadius: 6,
                        border: '1px solid #f59e0b', background: 'rgba(245,158,11,0.15)',
                        color: '#f59e0b', fontSize: 9, fontWeight: 700,
                        cursor: 'pointer', whiteSpace: 'nowrap',
                      }}>AI:{session.detectedCapo}</button>
                  )}
                </div>
                <span style={{ fontSize: 9, color: 'var(--st-text-dim)', fontWeight: 'bold' }}>カポ設定</span>
              </div>

              <div className="ribbon-divider" />

              {/* ⑤ Transpose */}
              <div className="ribbon-item">
                <div className="transpose-controls" style={{ gap: 2, marginBottom: 2 }}>
                  <button className="transpose-btn" onClick={() => { setTranspose(t => t - 1); _showToast('移調 −1'); }} style={{ width: 20, height: 20, fontSize: 12 }}>−</button>
                  <span className="transpose-label" style={{ fontSize: 11, minWidth: 24 }}>{transpose >= 0 ? '+' : ''}{transpose}</span>
                  <button className="transpose-btn" onClick={() => { setTranspose(t => t + 1); _showToast('移調 +1'); }} style={{ width: 20, height: 20, fontSize: 12 }}>+</button>
                </div>
                <span style={{ fontSize: 9, color: 'var(--st-text-dim)', fontWeight: 'bold' }}>転調</span>
              </div>

              <div className="ribbon-divider" />

              {/* ⭐ Metronome */}
              <div className="ribbon-item">
                <button
                  onClick={() => setMetronomeEnabled(e => !e)}
                  className={`ribbon-btn ${metronomeEnabled ? 'active' : ''}`}
                  title="メトロノームON/OFF"
                >
                  <Timer size={18} />
                  <span style={{ fontSize: 9, marginTop: 4, fontWeight: 'bold' }}>クリック</span>
                </button>
              </div>

              <div className="ribbon-divider" />

              {/* ⭐ Sync Offset */}
              <div className="ribbon-item" title="音源と譜面（クリック）のタイミング微調整 (ミリ秒)">
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 2 }}>
                  <input
                    type="range" min="-500" max="500" step="10"
                    value={syncOffset}
                    onChange={(e) => setSyncOffset(Number(e.target.value))}
                    style={{ width: 64, accentColor: 'var(--st-brand)', cursor: 'pointer' }}
                  />
                  <span style={{ fontSize: 11, minWidth: 28, textAlign: 'right' }}>{syncOffset > 0 ? '+' : ''}{syncOffset}</span>
                </div>
                <span style={{ fontSize: 9, color: 'var(--st-text-dim)', fontWeight: 'bold' }}>同期補正(ms)</span>
              </div>

              <div className="ribbon-divider" />

              {/* ⭐ Tempo Multiplier */}
              <div className="ribbon-item" title="長時間の再生で生じるズレ（ドリフト）を補正する倍率">
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 2 }}>
                  <input
                    type="range" min="0.9500" max="1.0500" step="0.0001"
                    value={tempoMultiplier}
                    onChange={(e) => setTempoMultiplier(Number(e.target.value))}
                    style={{ width: 64, accentColor: 'var(--st-brand)', cursor: 'pointer' }}
                  />
                  <span style={{ fontSize: 11, minWidth: 32, textAlign: 'right' }}>{tempoMultiplier.toFixed(4)}x</span>
                </div>
                <span style={{ fontSize: 9, color: 'var(--st-text-dim)', fontWeight: 'bold' }}>テンポ補正</span>
              </div>

              <div className="ribbon-divider" />

              {/* ⑥ Noise Gate */}
              <div className="ribbon-item" title="AIのノイズ除去レベル。右にするほど低velocity音が消えてシンプルになります">
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 2 }}>
                  <input
                    type="range" min="0" max="0.8" step="0.05"
                    value={noiseGate}
                    onChange={(e) => setNoiseGate(parseFloat(e.target.value))}
                    style={{ width: '80px', accentColor: '#4da6ff', cursor: 'pointer' }}
                  />
                  <button
                    onClick={async () => {
                      if (!session?.id) return;
                      handleRetune(null, null, noiseGate);
                    }}
                    style={{
                      padding: '2px 6px', borderRadius: 4, border: 'none',
                      background: noiseGate > 0 ? '#4da6ff' : '#334155',
                      color: 'white', fontSize: 10, fontWeight: 700,
                      cursor: 'pointer', whiteSpace: 'nowrap',
                    }}
                  >適用</button>
                </div>
                <span style={{ fontSize: 9, color: 'var(--st-text-dim)', fontWeight: 'bold' }}>ノイズ除去 (CUT: {Math.round(noiseGate * 100)}%)</span>
              </div>

              {/* ⑦ Writes / Exports (Aligned Right) */}
              <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginLeft: 'auto' }}>
                {/* PDF */}
                <div className="ribbon-item">
                  <button 
                    className="ribbon-action-btn"
                    onClick={() => window.open(`${API_BASE}/result/${session.id}/pdf`, '_blank')}
                    title="PDFでTAB譜を印刷または保存します"
                  >
                    <Printer size={16} />
                    <span>PDF</span>
                  </button>
                  <span className="ribbon-label">PDF印刷</span>
                </div>

                <div className="ribbon-divider" />

                {/* Guitar Pro */}
                <div className="ribbon-item">
                  <button 
                    className="ribbon-action-btn gp5"
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
                    title="Guitar Pro用ファイル (.gp5) をダウンロードします"
                  >
                    <Download size={16} />
                    <span>GP5</span>
                  </button>
                  <span className="ribbon-label">Guitar Pro</span>
                </div>

                <div className="ribbon-divider" />

                {/* MusicXML */}
                <div className="ribbon-item">
                  <button 
                    className="ribbon-action-btn"
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
                    title="MusicXMLファイルをダウンロードします"
                  >
                    <Download size={16} />
                    <span>XML</span>
                  </button>
                  <span className="ribbon-label">MusicXML</span>
                </div>

                <div className="ribbon-divider" />

                {/* TuxGuitar */}
                <div className="ribbon-item">
                  <button 
                    className="ribbon-action-btn tux"
                    onClick={async () => {
                      const isLocal = API_BASE.includes('localhost') || API_BASE.includes('127.0.0.1');
                      if (isLocal) {
                        try {
                          const res = await fetch(`${API_BASE}/result/${session.id}/open-tuxguitar`, { method: 'POST' });
                          if (!res.ok) {
                            const err = await res.json().catch(() => ({}));
                            throw new Error(err.detail || "起動失敗");
                          }
                          _showToast("TuxGuitarを起動しました");
                        } catch(e) { _showToast("TuxGuitar: " + e.message); }
                      } else {
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
                          _showToast("GP5をダウンロードしました");
                        } catch(e) { _showToast("GP5: " + e.message); }
                      }
                    }}
                    title="ローカルのTuxGuitarを起動、またはGP5をダウンロードします"
                  >
                    <Music size={16} />
                    <span>Tux</span>
                  </button>
                  <span className="ribbon-label">TuxGuitar</span>
                </div>
              </div>
            </div>

            {/* TAB View */}
            <div className="tab-container" style={{ paddingBottom: 0 }}>
              <TabView
                key={retuneKey}
                sessionId={session.id}
                apiBase={API_BASE}
                currentTime={currentTime}
                isPlaying={isPlaying || scrollOnly}
                transpose={transpose}
                capo={capo}
                metronomeEnabled={metronomeEnabled}
                syncOffset={syncOffset}
                tempoMultiplier={tempoMultiplier}
                onNoteEdited={() => setRetuneKey(k => k + 1)}
              />
            </div>
          </>
        )}
      </div>

      {toast && <div className="toast">{toast}</div>}
      <footer style={{ padding: '12px 0', textAlign: 'center', fontSize: '0.7rem', color: 'rgba(255,255,255,0.25)', borderTop: '1px solid rgba(255,255,255,0.06)', marginTop: 'auto' }}>
        © 2026 <a href="https://baseline-designs.com" target="_blank" rel="noopener noreferrer" style={{ color: 'inherit', textDecoration: 'none' }} className="hover:underline">BaseLineDesigns Inc.</a> All rights reserved.
      </footer>
    </div>
  );
}

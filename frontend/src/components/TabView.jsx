import React, { useEffect, useState } from "react";

/**
 * TabView — GP5ダウンロード & 楽譜情報表示
 * AlphaTab完全排除。GP5はPower Tab Editor / TuxGuitar / Guitar Proで開く。
 */
const TabViewInner = ({ sessionId, apiBase }) => {
    const [info, setInfo] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!sessionId) return;
        fetch(`${apiBase}/result/${sessionId}`)
            .then(r => r.json())
            .then(setInfo)
            .catch(e => setError(e.message));
    }, [sessionId, apiBase]);

    const gp5Url = `${apiBase}/result/${sessionId}/gp5`;

    // OSのファイル関連付けでPower Tab / TuxGuitarが自動起動
    const openInEditor = () => {
        window.open(gp5Url, "_blank");
    };

    const downloadGp5 = async () => {
        try {
            const res = await fetch(gp5Url);
            if (!res.ok) throw new Error("取得失敗");
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${(info?.filename || "tab").replace(/\.[^.]+$/, "")}.gp5`;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 200);
        } catch (e) {
            alert("GP5ダウンロード失敗: " + e.message);
        }
    };

    if (error) {
        return (
            <div style={styles.container}>
                <div style={styles.card}>
                    <p style={{ color: "#ef4444" }}>❌ {error}</p>
                </div>
            </div>
        );
    }

    return (
        <div style={styles.container}>
            <div style={styles.card}>
                <div style={styles.icon}>🎸</div>
                <h2 style={styles.title}>TAB譜の準備ができました</h2>

                <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
                    <button onClick={openInEditor} style={styles.openBtn}>
                        🎵 Power Tab で開く
                    </button>
                    <button onClick={downloadGp5} style={styles.downloadBtn}>
                        ⬇ GP5 保存
                    </button>
                </div>

                {info && (
                    <div style={styles.meta}>
                        {info.bpm && <span style={styles.badge}>♩ {Math.round(info.bpm)} BPM</span>}
                        {info.total_notes && <span style={styles.badge}>♪ {info.total_notes} notes</span>}
                        {info.key && <span style={styles.badge}>🎵 {info.key}</span>}
                        {info.capo > 0 && <span style={styles.badge}>Capo {info.capo}</span>}
                    </div>
                )}

                <p style={styles.hint}>
                    💡 Power Tab Editor (無料) は <a href="https://www.power-tab.net/" target="_blank" rel="noreferrer" style={styles.link}>power-tab.net</a> からダウンロードできます
                </p>
            </div>
        </div>
    );
};

const styles = {
    container: {
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100%",
        padding: "40px 20px",
    },
    card: {
        background: "var(--st-surface, #1e1e22)",
        border: "1px solid var(--st-border, #333)",
        borderRadius: 16,
        padding: "48px 40px",
        textAlign: "center",
        maxWidth: 480,
        width: "100%",
        boxShadow: "0 8px 32px rgba(0,0,0,0.3)",
    },
    icon: { fontSize: 56, marginBottom: 16 },
    title: {
        fontSize: 20,
        fontWeight: 700,
        margin: "0 0 12px",
        color: "var(--st-text, #e0e0e0)",
    },
    subtitle: {
        fontSize: 14,
        color: "var(--st-text-dim, #999)",
        lineHeight: 1.6,
        margin: "0 0 24px",
    },
    openBtn: {
        background: "linear-gradient(135deg, #f59e0b, #fb923c)",
        color: "#000",
        border: "none",
        borderRadius: 10,
        padding: "14px 36px",
        fontSize: 16,
        fontWeight: 700,
        cursor: "pointer",
        boxShadow: "0 4px 16px rgba(245,158,11,0.3)",
    },
    downloadBtn: {
        background: "transparent",
        color: "var(--st-text-dim, #aaa)",
        border: "1px solid var(--st-border, #444)",
        borderRadius: 10,
        padding: "14px 24px",
        fontSize: 14,
        fontWeight: 600,
        cursor: "pointer",
    },
    meta: {
        display: "flex",
        gap: 8,
        justifyContent: "center",
        flexWrap: "wrap",
        marginTop: 24,
    },
    badge: {
        background: "var(--st-surface-3, #2a2a2e)",
        padding: "4px 12px",
        borderRadius: 20,
        fontSize: 12,
        color: "var(--st-text-dim, #aaa)",
    },
    hint: {
        marginTop: 24,
        fontSize: 12,
        color: "var(--st-text-dim, #777)",
    },
    link: { color: "#f59e0b" },
};

// Wrapper (keeps same export interface as before)
export const TabView = (props) => <TabViewInner {...props} />;
export default TabView;

import sys
import subprocess
import threading
import time
import os
import signal
from pathlib import Path

def tail_process(process, prefix):
    for line in iter(process.stdout.readline, b''):
        try:
            try:
                text = line.decode('utf-8').rstrip()
            except UnicodeDecodeError:
                text = line.decode('cp932', errors='replace').rstrip()
            if text:
                print(f"{prefix} {text}", flush=True)
        except Exception:
            pass

def _kill_port(port: int):
    """指定ポートをリッスンしている全プロセスを確実にキルする"""
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                if parts:
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) > 0:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True, timeout=5
                        )
                        print(f"  [cleanup] ポート {port} 上のプロセス (PID: {pid}) を強制終了しました", flush=True)
    except Exception as e:
        print(f"  Port {port} cleanup skipped: {e}", flush=True)

def main():
    print("===================================================", flush=True)
    print(" SoloTab - 強力リフレッシュ サーバー起動マネージャー", flush=True)
    print("===================================================", flush=True)
    
    # --- ゾンビプロセスの確実な掃除 ---
    print("[cleanup] 残存している古いプロセスを完全終了中...")
    for port in [8000, 8001, 8002, 5173, 5174, 5175]:
        _kill_port(port)
    
    PROJECT_ROOT = Path(__file__).resolve().parent

    # 前回セッションの中間生成ファイルを削除
    uploads_dir = PROJECT_ROOT / "uploads"
    if uploads_dir.exists():
        stale_files = ["tab_dual.musicxml", "tab.pdf"]
        stale_count = 0
        for session_dir in uploads_dir.iterdir():
            if not session_dir.is_dir():
                continue
            for fname in stale_files:
                f = session_dir / fname
                if f.exists():
                    try:
                        f.unlink()
                        stale_count += 1
                    except Exception:
                        pass
        if stale_count:
            print(f"[cleanup] 前回の中間キャッシュファイル x{stale_count} をクリーンアップしました")

    # システムの Python 3.13 / 仮想環境の検出
    python_bin = sys.executable or "python"

    backend_cmd = [
        python_bin,
        "-u", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000",
    ]
    
    frontend_env = os.environ.copy()
    frontend_env["CI"] = "true"  
    
    backend_env = os.environ.copy()
    backend_env["PYTHONUNBUFFERED"] = "1"
    backend_env["PYTHONIOENCODING"] = "utf-8"
    backend_env["PYTHONUTF8"] = "1"
    backend_env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    backend_env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    backend_env["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::DeprecationWarning"

    # Windows では npm.cmd を明示的に指定
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    
    print("[1/2] バックエンドを起動中... (Port 8000)", flush=True)
    p_backend = subprocess.Popen(
        backend_cmd,
        cwd=str(PROJECT_ROOT / "backend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=backend_env,
        bufsize=0
    )
    
    print("[2/2] フロントエンドを起動中... (Port 5174)", flush=True)
    p_frontend = subprocess.Popen(
        frontend_cmd,
        cwd=str(PROJECT_ROOT / "frontend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=frontend_env,
        bufsize=0
    )
    
    t_backend = threading.Thread(target=tail_process, args=(p_backend, "[BACKEND] "))
    t_frontend = threading.Thread(target=tail_process, args=(p_frontend, "[FRONTEND]"))
    
    t_backend.daemon = True
    t_frontend.daemon = True
    t_backend.start()
    t_frontend.start()
    
    print("\n===================================================")
    print(">>> 起動完了！最新コードで正常稼働中")
    print(">>> バックエンド:   http://localhost:8000")
    print(">>> フロントエンド: http://localhost:5174")
    print(">>> 終了時は [Ctrl+C] を押してください。")
    print("===================================================\n")
    
    try:
        while True:
            time.sleep(1)
            if p_backend.poll() is not None:
                print("\n[!] Backend が終了しました。")
                break
            if p_frontend.poll() is not None:
                print("\n[!] Frontend が終了しました。")
                break
    except KeyboardInterrupt:
        print("\n[shutdown] サーバーを終了しています...")
    finally:
        # プロセスを確実に道連れ終了
        try:
            p_backend.terminate()
            p_frontend.terminate()
            time.sleep(0.5)
            p_backend.kill()
            p_frontend.kill()
        except Exception:
            pass
        for port in [8000, 5174]:
            _kill_port(port)
        print("[shutdown] 全プロセスをクリーンアップ完了しました。")

if __name__ == "__main__":
    main()

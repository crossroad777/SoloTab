import sys
import subprocess
import threading
import time
import os

def tail_process(process, prefix):
    # 文字化け防止のため utf-8 / replace でデコード
    for line in iter(process.stdout.readline, b''):
        try:
            print(f"{prefix} {line.decode('utf-8', errors='replace').rstrip()}")
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
                        print(f"  Killed PID {pid} on port {port}")
    except Exception as e:
        print(f"  Port {port} cleanup skipped: {e}")


def main():
    print("=======================================")
    print(" SoloTab - 統合一発起動スクリプト")
    print("=======================================")
    
    # --- ゾンビプロセスの確実な掃除 ---
    print("[cleanup] 残存プロセスを掃除中...")
    _kill_port(8001)
    _kill_port(5174)
    
    # __pycache__ を削除して古いバイトコードが残らないようにする
    import shutil
    from pathlib import Path
    cache_dir = Path(r"D:\Music\nextchord-solotab\backend\__pycache__")
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print("[cleanup] __pycache__ を削除しました")
        except Exception as e:
            print(f"[cleanup] __pycache__ 削除失敗: {e}")

    backend_cmd = [
        "D:\\Music\\nextchord\\venv312\\Scripts\\python.exe",
        "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001",
        "--reload", "--reload-dir", "."
    ]
    
    # [重要] $env:CI="true" を付与しないと Vite は非対話ターミナルですぐ死ぬ
    frontend_env = os.environ.copy()
    frontend_env["CI"] = "true"  
    
    frontend_cmd = ["cmd", "/c", "npm run dev"]
    
    # Windowsでプロセスグループを作ってCtrl+Cでまとめてキルしやすくするフラグ
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    
    print("[1/2] Starting Backend...  (Port 8001)")
    p_backend = subprocess.Popen(
        backend_cmd,
        cwd="D:\\Music\\nextchord-solotab\\backend",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1
    )
    
    print("[2/2] Starting Frontend... (Port 5174)")
    p_frontend = subprocess.Popen(
        frontend_cmd,
        cwd="D:\\Music\\nextchord-solotab\\frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=frontend_env,
        bufsize=1
    )
    
    # ログ出力用スレッド開始
    t_backend = threading.Thread(target=tail_process, args=(p_backend, "[BACKEND] "))
    t_frontend = threading.Thread(target=tail_process, args=(p_frontend, "[FRONTEND]"))
    
    t_backend.daemon = True
    t_frontend.daemon = True
    t_backend.start()
    t_frontend.start()
    
    print("\n>>> Done! \n>>> Backend: http://localhost:8001 \n>>> Frontend: http://localhost:5174")
    print(">>> 終了時は [Ctrl+C] を押してください。\n")
    
    try:
        while True:
            time.sleep(1)
            # もしどちらかが死んだ場合のエラーハンドリング
            if p_backend.poll() is not None:
                print("=======================================")
                print("[!] Backend が予期せず終了しました。")
                print("=======================================")
                break
            if p_frontend.poll() is not None:
                print("=======================================")
                print("[!] Frontend が予期せず終了しました。")
                print("=======================================")
                break
    except KeyboardInterrupt:
        print("\n[Ctrl+C] シャットダウンシグナルを受け取りました...")
    finally:
        # 子プロセスのクリーンキル (タスクマネージャ送り対策)
        print("プロセスを停止中...")
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p_backend.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p_frontend.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        print("終了しました。")
        sys.exit(0)

if __name__ == "__main__":
    main()

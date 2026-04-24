---
description: SoloTabアプリケーションを起動する
---

# SoloTab 起動手順

// turbo-all

1. 起動スクリプトを実行する:
```powershell
& "D:\Music\nextchord-solotab\start.ps1"
```

このスクリプトが自動的に以下を行います:
- 古いプロセスの停止（ポート8001の競合解消）
- バックエンド起動 (port 8001) + 起動完了まで待機
- フロントエンド起動 (port 5174)

2. ブラウザで http://localhost:5174/ を開く

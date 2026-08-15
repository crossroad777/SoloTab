# SoloTab プロジェクト引継書 (Handover Document)
**作成日時**: 2026-08-16 | **フェーズ**: Phase 8 完了時点（Universal Quantizer 実装・実音源アルペジオ検証完了）

---

## 1. プロジェクト概要と最新ステータス

### 1.1 プロジェクトの目的
ソロギター・フィンガースタイルギターのための、実環境（マイク録音・ノイズ・ボディ共鳴あり）で高精度に動作する完全自動採譜・TAB譜生成システム（SoloTab）の開発。

### 1.2 現在のステータス（Phase 8 完了）
- **ピッチ認識精度 (Pitch F1)**: **`0.8611`** (Precision: 0.8578, Recall: 0.8655)
- **弦割り当て精度 (String Accuracy)**: **`82.05%`** (実用基準 82.0% 達成)
- **見逃しエラー (A1 Error)**: **`105個`** (Phase 2の149個から大幅削減・高Recall維持)
- **過剰検出エラー (A2 Error)**: **`177個`** (Phase 5の224個から -47個 削減、目標 `<= 180` 達成)
- **音高不一致エラー (A3 Error)**: **`190個`** (目標 `<= 190` 達成)
- **汎用量化エンジン (Universal Quantizer)**: **実装完了**
  - プロファイル・奏法別モードに頼らず、BPM・拍子から数学的グリッド（Straight 16分 vs 8分3連符 vs 16分3連符/6連符等）を拍ごとに二乗誤差で自動最適スナップ。
  - 実音源「禁じられた遊び（`romance.wav`）」において、Standard Mode のままで全小節の8分3連符アルペジオ（1拍3音）を正確に検出し、MusicXML / GP5 上で3連符記号「3」（Tuplet Bracket）で美しく束ねることに成功。
- **本番デプロイ**: Vercel 本番稼働中 (`https://solotab.vercel.app`)
- **ローカル起動環境**: ワンクリック起動バッチ [`quick_start.bat`](file:///d:/Music/chordlink-solotab/quick_start.bat) 完備（フロントエンド 5174ポート、バックエンド 8002ポート）
- **Git同期**: GitHub リモート / ローカル Git コミット完了 (`commit: 1954923`)
- **学術論文ドラフト**: [`docs/academic_paper_draft.md`](file:///d:/Music/chordlink-solotab/docs/academic_paper_draft.md) 格納済み

---

## 2. フェーズ別 成果と性能推移

| 評価指標 | Phase 2 (旧ベースライン) | Phase 3 (全7モデル再学習) | Phase 5 (DP・閾値最適化) | Phase 6.5 (高精度達成) | Phase 8 (Universal Quantizer) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pitch F1 Score** | 0.8252 | 0.8348 | 0.8599 | **`0.8611`** | **`0.8611`** 🏆 |
| **String Accuracy** | 77.91% | 75.03% | 81.92% | **`82.05%`** | **`82.05%`** 🎯 |
| **A1 エラー (見逃し)** | 149個 | 87個 | 87個 | **`105個`** | **`105個`** 🏆 |
| **A2 エラー (過剰検出)** | 195個 | 238個 | 224個 | **`177個`** | **`177個`** 🏆 |
| **A3 エラー (音高不一致)** | 185個 | 197個 | 195個 | **`190個`** | **`190個`** ✅ |
| **アルペジオ・3連符記譜** | 16分に歪む / 記号欠落 | - | - | 奏法プロファイル必要 | **完全自動・数理的スナップ** ✨ |

---

## 3. 主要な構造改革とブレークスルー

### ① Universal Quantizer（汎用量化エンジン）
- **背景**: 奏法別プロファイル（Classic Mode等）によるアドホックな対応を排し、AIは「物理的時間（ミリ秒）」の計算機に徹する方針へ転換。
- **解決**: `DIVISIONS=12` を基準単位とし、各拍内のオンセットクラスタに対して Straight vs 8th Triplet vs 16th Triplet の誤差を計算。3分割パターン（0, 4, 8 divs）を自動グルーピングし、MusicXML の `<tuplet type="start" bracket="yes"/>` / `<tuplet type="stop"/>` および GP5 の `gp.Tuplet(3, 2)` を全小節で一貫して正確に出力。

### ② The Silent Dropout（データローダーの波形スキップバグ根絶）
- **解決**: `.pt` CQTキャッシュのフォールバック機構を導入し、556トラック全量を投入して全7ドメインモデル（MoE）を完全再学習。A1エラー（見逃し）が 149個 $\rightarrow$ 87個（41.6%）激減。

### ③ The Frozen Randomness（乱数シード固定バグ根絶）
- **解決**: エポックごとの動的シード再生成と Gradient Clipping（1.0）を導入し、真の汎化性能を獲得。

### ④ The Viterbi Override（CNNハードプロテクト）
- **解決**: CNN確信度 $\ge 0.90$ で Viterbi 遷移コストを無効化する「CNNハードプロテクト」を実装。String Accuracy が 75% $\rightarrow$ **82.05%** に飛躍。

### ⑤ 音楽理論・倍音構造フィルタ（Pass 2 v3）
- **解決**: 同時発音グループ（$\Delta t \le 50$ms）内で、基本音より弱くコード外の高次倍音（オクターブ・5度）を自動除去。A2エラー（過剰検出）を 224個 $\rightarrow$ 177個（-47個）削減。

---

## 4. 主要ファイル構成と役割

```
D:\Music\chordlink-solotab\
├── backend/
│   ├── pipeline.py                 # E2Eメインパイプライン（BP+MoE融合、UQ統合）
│   ├── universal_quantizer.py       # 汎用数学的量子化エンジン（Phase 8）
│   ├── tab_renderer.py             # MusicXML TAB譜レンダラー（全小節Tuplet対応）
│   ├── gp_renderer.py              # Guitar Pro 5 (.gp5) レンダラー（AlphaTab完全互換）
│   ├── music_theory.py             # 音楽理論エンジン（MVSスコア、倍音フィルタ）
│   ├── string_assigner.py          # 弦・運指決定エンジン（CNNハードプロテクト、Minimax Viterbi）
│   ├── chord_theory.py             # 和音理論・コード補正エンジン
│   ├── models_fast/                # 再学習済み全7ドメイン MoE CRNN モデル（.pth）
│   └── benchmark/
│       ├── verify_universal_quantizer.py # UQアルペジオ3連符検証スクリプト
│       ├── e2e_pipeline_benchmark.py     # GuitarSet 9曲 E2Eベンチマーク
│       └── mini_benchmark.py             # 10曲回帰テスト
├── docs/
│   ├── handover_document.md        # 本引継書
│   └── academic_paper_draft.md     # 学術論文ドラフト
├── frontend/                       # Vite + React Webフロントエンド（Vercel本番稼働）
└── quick_start.bat                 # ワンクリック起動バッチ
```

---

## 5. 次のステップ（Next Actions）

1. **実機テスト（リアルギター音声の検証継続）**:
   - ユーザーの実機ギター音源（アコギ・クラシック・エレキ）をアップロードし、TAB譜と3連符・アルペジオ・コードストロークの視認性を確認。
2. **Web UI上でのブラッシュアップ**:
   - AlphaTab 上での3連符表示や再生同期の微細な調整。
3. **論文ドラフト（`academic_paper_draft.md`）の更新**:
   - Universal Quantizer の数理モデルと実音源検証結果を反映。

---

## 6. 新しい会話を開始する際のおすすめプロンプト

新しいセッションを開始する際は、以下のテキストをコピー＆ペーストしてチャットに送信してください：

```markdown
SoloTabプロジェクトの作業を再開します。
引き継ぎドキュメント（docs/handover_document.md）の内容を読み込んでください。

【現在の状態】
- Phase 8 完了（Pitch F1=0.8611, String Accuracy=82.05%, Universal Quantizer 実装・検証完了）
- 実音源「禁じられた遊び」にて Standard Mode での 8分3連符アルペジオ自動量化・Tuplet記譜に成功
- Vercel本番デプロイ済み（https://solotab.vercel.app）、Git同期済み
- ワンクリック起動バッチ quick_start.bat 完備

【今回の目的】
実機テストおよびUI/UXの最終確認、論文ドラフトの更新を進めたいと思います。
準備ができたら教えてください。
```

# SoloTab Legacy Logs Archive & Metadata Registry

本ディレクトリ（`_archive/legacy_logs/`）は、SoloTab v2 の研究開発過程（論文 `solotab_v2_sota_paper_ja.md` §8〜§20）で記録された実験・学習・ベンチマークの実行ログファイル群（全51件）を安全に隔離・保管したアーカイブです。

---

## ■ ログファイル分類一覧

### 1. MoE エキスパートモデル訓練ログ（§10）
- `train_6_experts.log` / `train_6_experts_day2.log` — 6つの MoE 分野別専門家モデルの訓練損失推移ログ
- `train_martin_finger.log` — Martin アコースティックギター特化指番号モデルの訓練ログ

### 2. ベンチマーク・精度評価ログ（§11〜§13）
- `synth_v2_moe_benchmark_results.log` — 合成データセット v2 での MoE 各モデル Precision/Recall/F1 評価ログ
- `moe_benchmark_final.log` — 最終 MoE アンサンブル性能評価ログ
- `final_mini.log` / `final_mini_vote6_correct.log` — ミニベンチマーク（10曲）の投票閾値別 F1 検証ログ
- `final_day3_benchmark.log` / `day3_vote6_mini.log` / `day3_vote6_mini_fixed.log` — Day 3 最適化時の検証ログ
- `tune1_mini.log` / `tune5_mini.log` — チューニング別（変則/レギュラー）ミニベンチマークログ
- `benchmark_baseline_sota.log` / `benchmark_progress.log` / `benchmark_error.log` — SOTA 比較ベンチマークログ

### 3. ドメイン間ミスマッチ・ピッチ誤差解析ログ（§12）
- `mismatch_jazz.log` / `mismatch_jazz_tune1~5.log` — ジャズ音源におけるドメイン適応とチューニング誤認識ログ
- `mismatch_rock1.log` / `mismatch_rock1_pitch.log` / `mismatch_rock2.log` — ロック/エレキギター音源での倍音・歪み誤差解析ログ
- `pitch_error_1domain.log` / `pitch_error_adaptive.log` — 単一ドメイン vs 適応型ドメインでのピッチ誤差比較ログ

### 4. パイプライン復活・自動化・高速化実験ログ（§14〜§16）
- `resurrection_automated.log` / `resurrection_automated_v4~v6.log` — パイプライン自動検証ログ
- `resurrection_progress.log` / `resurrection_progress_v2~v3.log` / `resurrection_final.log` — 復元進行ログ
- `resurrection_accelerated_v7.log` / `acceleration_test.log` — GPU 高速推論・並列化ログ
- `resurrection_revert_stable.log` — 安定版ロールバック検証ログ

### 5. E2E ドライラン・本番動作ログ（§14）
- `e2e_out_dryrun.log` / `e2e_out_dryrun2~4.log` — 初期 E2E パイプラインのドライラン実行ログ
- `e2e_out_adaptive.log` / `e2e_out_phase3.log` — 適応型パラメータおよび Phase 3 パイプライン実行ログ
- `e2e_out_prod.log` / `e2e_out_prod_fallback.log` — 本番フォールバック動作検証ログ
- `e2e_1domain.log` / `final_phase3.log` — 最終統合テスト実行ログ
- `pipeline_error.log` / `error_traceback.log` / `server.log` / `run_all_domains_moe.log` — サーバー実行時デバッグログ

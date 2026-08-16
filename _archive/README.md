# SoloTab Legacy Experiments Archive & Metadata Registry

本ディレクトリ（`_archive/legacy_experiments/`）は、SoloTab v2 の研究開発過程（論文 `solotab_v2_sota_paper_ja.md` §8〜§20）において実施された実験スクリプト、モデル学習コード、および検証ツールのアーカイブです。
プロダクション・クリティカルパス（聖域）から安全に隔離され、将来の再検証やデータセット再構築のために永続保存されています。

---

## ■ 実験スクリプト vs 論文セクション対応メタデータ

| ファイル名 | 関連論文セクション | 実験目的・概要 | 結論・採否 |
|---|---|---|---|
| `train_finger_cnn_26k.py` | §10.4 Step 3 / §18.2 | SoloTab-26K データセットを用いた左手指番号CNNの訓練 | **採用（成功）**: 指番号推定精度 84.1% 達成 |
| `train_noise_robust_solotab26k.py` | §8.6.4 / §10.12 | ノイズ重畳合成データによる頑健性向上訓練 | **不採用（失敗）**: 実音源でのF1が低下、生データ優先方針へ |
| `train_technique_cnn.py` | §15.2 / §20.3 | スラー・チョーキング・ハーモニクス等の奏法識別モデル訓練 | **採用（成功）**: 奏法分類器 v4 のベースライン確立 |
| `train_technique_cnn_v4_finetune.py` | §15.4 / §20.4 | 奏法識別 CNN の微細調整（Fine-tuning） | **採用（成功）**: ピックアップ・ハンマリング検出精度向上 |
| `train_technique_head.py` | §15.3 | 奏法分類ヘッドの重み最適化 | **完了**: 各種アテンション重みの探索完了 |
| `optuna_finger_weights.py` | §19.4 | Optuna による Viterbi DP 運指遷移コスト重みのベイズ最適化 | **完了**: `w_movement=25.0, w_span=10.0` の最適値を導出 |
| `guitar_cost_functions_baseline_phase2.py` | §18.3 | Phase 2 運指コスト関数の旧ベースライン実装 | **アーカイブ（旧版）**: `guitar_cost_functions.py` へ統合 |
| `music_quantizer.py` / `music_quantizer_debug.py` | §6.0 | 拍・小節への過度な音符量子化実験 | **不採用（廃止）**: アルペジオの微細ニュアンスを破壊するため廃止 |
| `patch_quantizer.py` / `patch_tabview.py` | §17.3 | 初期フロントエンドとレンダラーのパッチ適用検証 | **完了（ワンオフ）**: 修正が本線コードに統合されたため保存 |
| `musescore_renderer.py` | §7.2 | MuseScore による MusicXML レンダリング検証 | **アーカイブ**: PyGuitarPro (GP5) レンダラーを SSOT として統一 |
| `fingering_model.py` | §18.1 | 旧世代（v1）の運指予測モデル | **アーカイブ**: Viterbi DP + CNN-first へ移行 |
| `overnight_training.py` | §10.5 | 長時間バッチ学習用スケジューラ | **完了（ワンオフ）**: 実験完了 |
| `scrape_gprotab.py` | §8.2 | ギタープロ譜面データセット収集スクリプト | **完了（資産化）**: SoloTab-26K コーパスの構築完了 |
| `test_all_techniques_gp5.py` | §15.5 | GP5 エクスポート時の全奏法レンダリング検証 | **完了**: 各種テクニックの GP5 表現を実証 |
| `test_edge_cases.py` / `test_phase_d.py` | §16.2 | 限界音域・特殊変則チューニングの単体テスト | **完了**: 単体テストハーネスの旧版 |
| `generate_hp_test_phrases.py` | §14.2 | ハイポジション・難関フレーズの合成生成ツール | **完了**: 運指ストレステスト用フレーズ生成完了 |
| `check_json.py` / `check_notes.py` 等 | §8.4 / §12.1 | 中間 JSON・MIDI データ構造の目視・整合性確認スクリプト群 | **完了（ワンオフ）**: 開発時デバッグツール |
| `archive_cleanup.py` | — | リポジトリ整理スクリプト | **アーカイブ**: 役目完了 |

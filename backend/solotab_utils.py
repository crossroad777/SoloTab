import numpy as np

def _to_native(obj):
    """numpy型をPythonネイティブ型に再帰変換 (JSONシリアライズ用)"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# 標準チューニング: 6弦→1弦 (低→高)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4

# ===== ソロギター変則チューニング大辞典 =====
TUNINGS = {
    # --- スタンダード系 ---
    "standard":       [40, 45, 50, 55, 59, 64],  # E A D G B E
    "half_down":      [39, 44, 49, 54, 58, 63],  # Eb Ab Db Gb Bb Eb
    "full_down":      [38, 43, 48, 53, 57, 62],  # D G C F A D
    "1half_down":     [37, 42, 47, 52, 56, 61],  # Db Gb B E Ab Db

    # --- Drop系 ---
    "drop_d":         [38, 45, 50, 55, 59, 64],  # D A D G B E
    "drop_c#":        [37, 44, 49, 54, 58, 63],  # C# Ab Db Gb Bb Eb
    "drop_c":         [36, 43, 48, 53, 57, 62],  # C G C F A D
    "drop_b":         [35, 42, 47, 52, 56, 61],  # B Gb B E Ab Db
    "double_drop_d":  [38, 45, 50, 55, 59, 62],  # D A D G B D

    # --- DADGAD系 (ケルティック・押尾コータロー頻用) ---
    "dadgad":         [38, 45, 50, 55, 57, 62],  # D A D G A D
    "dadgac":         [38, 45, 50, 55, 57, 60],  # D A D G A C (押尾「twilight」等)
    "dadgae":         [38, 45, 50, 55, 57, 64],  # D A D G A E
    "dadead":         [38, 45, 50, 52, 57, 62],  # D A D E A D
    "cgdgad":         [36, 43, 50, 55, 57, 62],  # C G D G A D
    "cgdgbd":         [36, 43, 50, 55, 59, 62],  # C G D G B D
    "dadf#ad":        [38, 45, 50, 54, 57, 62],  # D A D F# A D (Open D variant)
    "daddad":         [38, 45, 50, 50, 57, 62],  # D A D D A D

    # --- Open Major系 ---
    "open_d":         [38, 45, 50, 54, 57, 62],  # D A D F# A D
    "open_e":         [40, 47, 52, 56, 59, 64],  # E B E G# B E
    "open_g":         [38, 43, 50, 55, 59, 62],  # D G D G B D
    "open_a":         [40, 45, 52, 57, 61, 64],  # E A E A C# E
    "open_c":         [36, 43, 48, 55, 60, 64],  # C G C G C E
    "open_c6":        [36, 45, 48, 55, 60, 64],  # C A C G C E

    # --- Open Minor系 ---
    "open_dm":        [38, 45, 50, 53, 57, 62],  # D A D F A D
    "open_em":        [40, 47, 52, 55, 59, 64],  # E B E G B E
    "open_gm":        [38, 43, 50, 55, 58, 62],  # D G D G Bb D
    "open_am":        [40, 45, 52, 57, 60, 64],  # E A E A C E
    "open_cm":        [36, 43, 48, 55, 60, 63],  # C G C G C Eb

    # --- Nashville / New Standard ---
    "nashville":      [52, 57, 62, 67, 71, 76],  # E3 A3 D4 G4 B4 E5 (高音)
    "new_standard":   [36, 43, 50, 57, 62, 67],  # C G D A E B (Fripp)

    # --- 押尾コータロー特有 ---
    "oshio_wind":     [38, 45, 50, 55, 57, 62],  # DADGAD (Wind Song等)
    "oshio_fight":    [38, 43, 50, 55, 59, 62],  # DGDGBD = Open G (Fight!等)
    "oshio_landscape":[36, 43, 50, 55, 57, 62],  # CGDGAD (Landscape等)

    # --- Andy McKee / Antoine Dufour / Michael Hedges ---
    "cgcgce":         [36, 43, 48, 55, 60, 64],  # C G C G C E = Open C
    "cgcgcg":         [36, 43, 48, 55, 60, 67],  # C G C G C G
    "bebebe":         [35, 40, 47, 52, 59, 64],  # B E B E B E
    "dadaad":         [38, 45, 50, 57, 57, 62],  # D A D A A D
    "cgdgbe":         [36, 43, 50, 55, 59, 64],  # C G D G B E

    # --- Eb系 (半音下げ) ---
    "eb_drop_db":     [37, 44, 49, 54, 58, 63],  # Db Ab Db Gb Bb Eb
}

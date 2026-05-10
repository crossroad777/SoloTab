"""
scrape_gprotab_stealth.py — GProTab.net ステルスGPファイル収集
================================================================
Playwrightベースのブラウザ自動化で、人間の行動パターンを再現。

ボット検知回避テクニック:
  - ランダム遅延（正規分布 + 時折の長い休憩）
  - マウスの揺らぎ移動（ベジェ曲線的な動き）
  - スクロール行動（人間はページを読む）
  - ランダムな訪問順序
  - セッション維持（Cookie保持）
  - リファラー自然遷移
  - 時折の「脱線」行動（関係ないページを見る）
  - User-Agent ローテーション

Usage:
    python backend/train/scrape_gprotab_stealth.py --letters a b --max-artists 20
    python backend/train/scrape_gprotab_stealth.py --all --max-artists 30
"""
import os, sys, time, random, json, re, math
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: pip install playwright && python -m playwright install chromium")
    sys.exit(1)

DOWNLOAD_DIR = Path(__file__).parent.parent.parent / "gprotab_downloads"
BASE_URL = "https://gprotab.net"
PROGRESS_FILE = DOWNLOAD_DIR / "progress_stealth.json"


# =============================================================================
# 人間の揺らぎ模倣
# =============================================================================

def human_delay(base=3.0, variance=2.0):
    """人間的なランダム遅延（正規分布）"""
    delay = max(1.0, random.gauss(base, variance))
    # 5%の確率で「コーヒー休憩」（長い停止）
    if random.random() < 0.05:
        coffee = random.uniform(8, 20)
        print(f"    [human] coffee break {coffee:.0f}s")
        delay += coffee
    time.sleep(delay)


def human_micro_delay():
    """マイクロ遅延（クリック前後の自然な間）"""
    time.sleep(random.uniform(0.3, 1.2))


def human_mouse_move(page, target_x=None, target_y=None):
    """人間的なマウス移動（直線ではなく揺らぎ付き）"""
    viewport = page.viewport_size
    if not viewport:
        return

    if target_x is None:
        target_x = random.randint(100, viewport["width"] - 100)
    if target_y is None:
        target_y = random.randint(100, viewport["height"] - 100)

    # 現在位置から目標まで数ステップで移動（揺らぎ付き）
    steps = random.randint(3, 8)
    current_x = random.randint(200, viewport["width"] - 200)
    current_y = random.randint(200, viewport["height"] - 200)

    for i in range(steps):
        progress = (i + 1) / steps
        # ベジェ曲線的な補間 + ランダム揺らぎ
        ease = progress * progress * (3 - 2 * progress)  # smoothstep
        x = int(current_x + (target_x - current_x) * ease + random.gauss(0, 5))
        y = int(current_y + (target_y - current_y) * ease + random.gauss(0, 3))
        x = max(0, min(viewport["width"], x))
        y = max(0, min(viewport["height"], y))

        page.mouse.move(x, y)
        time.sleep(random.uniform(0.02, 0.08))


def human_scroll(page):
    """人間的なスクロール（ページを読んでいるかのように）"""
    scroll_count = random.randint(1, 4)
    for _ in range(scroll_count):
        delta = random.randint(100, 400)
        if random.random() < 0.15:  # 15%で上にスクロール（読み返し）
            delta = -delta
        page.mouse.wheel(0, delta)
        time.sleep(random.uniform(0.5, 2.0))


def human_distraction(page):
    """時折の「脱線」行動 — 人間は脱線する"""
    if random.random() < 0.08:  # 8%の確率
        print(f"    [human] distraction: browsing random page...")
        try:
            # ロゴをクリックしてトップへ、少し眺めて戻る
            page.goto(BASE_URL + "/en", timeout=10000)
            human_delay(2, 1)
            human_scroll(page)
            human_delay(1, 0.5)
        except Exception:
            pass


# =============================================================================
# ブラウザ設定
# =============================================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

VIEWPORT_SIZES = [
    {"width": 1920, "height": 1080},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1366, "height": 768},
    {"width": 1280, "height": 720},
]


def create_stealth_browser(playwright):
    """ステルスブラウザを起動"""
    ua = random.choice(USER_AGENTS)
    vp = random.choice(VIEWPORT_SIZES)

    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ]
    )

    context = browser.new_context(
        user_agent=ua,
        viewport=vp,
        locale="en-US",
        timezone_id="America/New_York",
        ignore_https_errors=True,
        # WebDriver検出回避
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
        }
    )

    # navigator.webdriver を隠す
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => false });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        window.chrome = { runtime: {} };
    """)

    page = context.new_page()
    return browser, context, page


# =============================================================================
# スクレイピングロジック
# =============================================================================

def get_artists_for_letter(page, letter):
    """アーティスト一覧を取得"""
    url = f"{BASE_URL}/en/artists/{letter}"
    human_mouse_move(page)
    page.goto(url, timeout=15000)
    human_delay(2, 1)
    human_scroll(page)
    human_micro_delay()

    links = page.query_selector_all("a[href*='/en/tabs/']")
    artists = []
    for link in links:
        href = link.get_attribute("href") or ""
        text = link.inner_text().strip()
        if text and "/en/tabs/" in href:
            parts = href.strip("/").split("/")
            if len(parts) == 3:  # /en/tabs/artist
                artists.append((href, text))

    return list(set(artists))


def get_tabs_for_artist(page, artist_url, artist_name):
    """アーティストページからタブURL一覧を取得"""
    full_url = BASE_URL + artist_url if not artist_url.startswith("http") else artist_url

    human_mouse_move(page)
    human_micro_delay()
    page.goto(full_url, timeout=15000)
    human_delay(2, 1)
    human_scroll(page)

    links = page.query_selector_all("a[href*='/en/tabs/']")
    tabs = []
    for link in links:
        href = link.get_attribute("href") or ""
        text = link.inner_text().strip()
        parts = href.strip("/").split("/")
        if len(parts) >= 4 and text:  # /en/tabs/artist/song
            tabs.append((href, text))

    return list(set(tabs))


def download_tab_file(page, context, tab_url, artist_name):
    """タブのGPファイルをダウンロード（ブラウザ経由）"""
    full_url = BASE_URL + tab_url if not tab_url.startswith("http") else tab_url

    # ファイル名を生成
    parts = tab_url.strip("/").split("/")
    song_name = parts[-1] if len(parts) >= 2 else "unknown"
    safe_artist = re.sub(r'[^\w\-]', '_', artist_name)
    safe_song = re.sub(r'[^\w\-]', '_', song_name)

    artist_dir = DOWNLOAD_DIR / safe_artist
    artist_dir.mkdir(parents=True, exist_ok=True)

    # 既にダウンロード済みか確認
    existing = list(artist_dir.glob(f"{safe_song}.*"))
    if existing:
        return None

    # まずタブページを訪問（人間は曲ページを見てからDLする）
    human_mouse_move(page)
    human_micro_delay()
    page.goto(full_url, timeout=15000)
    human_delay(2, 1.5)

    # ページを少し読む
    human_scroll(page)
    human_delay(1, 0.5)

    # ダウンロードリンクを探す
    download_url = full_url + ("&" if "?" in full_url else "?") + "download"

    try:
        # ダウンロードイベントを待機
        with page.expect_download(timeout=10000) as download_info:
            page.goto(download_url, timeout=10000)

        download = download_info.value
        filepath = artist_dir / f"{safe_song}.gp5"
        download.save_as(str(filepath))
        return str(filepath)

    except Exception:
        # フォールバック: 直接HTTPでダウンロード試行
        try:
            response = page.request.get(download_url)
            if response.status == 200 and len(response.body()) > 100:
                filepath = artist_dir / f"{safe_song}.gp5"
                filepath.write_bytes(response.body())
                return str(filepath)
        except Exception:
            pass
        return None


# =============================================================================
# メイン
# =============================================================================

def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_artists": [], "total_files": 0, "total_notes_estimate": 0}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GProTab stealth scraper")
    parser.add_argument("--letters", nargs="*", help="Letters to crawl")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-artists", type=int, default=20)
    parser.add_argument("--max-tabs", type=int, default=10)
    args = parser.parse_args()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        letters = list("abcdefghijklmnopqrstuvwxyz")
    elif args.letters:
        letters = args.letters
    else:
        letters = ["a"]

    # ランダムな順序（ボットは順番にクロールする→人間はしない）
    random.shuffle(letters)

    progress = load_progress()
    total_downloaded = 0

    print(f"=== GProTab Stealth Scraper ===")
    print(f"Letters: {letters}")
    print(f"Max artists/letter: {args.max_artists}, Max tabs/artist: {args.max_tabs}")
    print(f"Download dir: {DOWNLOAD_DIR}")
    print(f"Already completed: {len(progress['completed_artists'])} artists")

    with sync_playwright() as pw:
        browser, context, page = create_stealth_browser(pw)

        try:
            # 最初にトップページを訪問（自然な流入）
            print(f"\n[init] Visiting homepage...")
            page.goto(BASE_URL + "/en", timeout=15000)
            human_delay(3, 1)
            human_scroll(page)

            for letter in letters:
                print(f"\n=== Letter: {letter.upper()} ===")
                try:
                    artists = get_artists_for_letter(page, letter)
                except Exception as e:
                    print(f"  ERROR listing artists for '{letter}': {e}")
                    continue
                random.shuffle(artists)  # ランダム順
                print(f"  Artists: {len(artists)}")

                for i, (artist_url, artist_name) in enumerate(artists[:args.max_artists]):
                    if artist_name in progress["completed_artists"]:
                        continue

                    print(f"  [{i+1}] {artist_name}")

                    try:
                        # 時折の脱線
                        human_distraction(page)

                        tabs = get_tabs_for_artist(page, artist_url, artist_name)
                        random.shuffle(tabs)  # ランダム順
                        print(f"      Tabs: {len(tabs)}")

                        artist_count = 0
                        for tab_url, tab_name in tabs[:args.max_tabs]:
                            try:
                                # 人間的な遅延
                                human_delay(3, 2)

                                filepath = download_tab_file(page, context, tab_url, artist_name)
                                if filepath:
                                    print(f"      OK: {tab_name}")
                                    artist_count += 1
                                    total_downloaded += 1
                                else:
                                    # 10%の確率でスキップ理由を変える（人間は全部DLしない）
                                    if random.random() < 0.1:
                                        print(f"      [human] skipping rest of this artist")
                                        break
                            except Exception as tab_err:
                                print(f"      ERROR downloading '{tab_name}': {tab_err}")
                                continue

                        progress["completed_artists"].append(artist_name)
                        progress["total_files"] += artist_count
                        save_progress(progress)

                    except Exception as artist_err:
                        print(f"    ERROR processing '{artist_name}': {artist_err}")
                        # ブラウザがクラッシュした可能性 → 再起動
                        try:
                            page.goto(BASE_URL + "/en", timeout=15000)
                        except Exception:
                            print("    [recovery] Browser crash detected, restarting...")
                            try: browser.close()
                            except Exception: pass
                            browser, context, page = create_stealth_browser(pw)
                            page.goto(BASE_URL + "/en", timeout=15000)
                            human_delay(3, 1)
                        continue

                    # アーティスト間の休憩（人間は一気にやらない）
                    if random.random() < 0.15:
                        rest = random.uniform(15, 45)
                        print(f"    [human] taking a break {rest:.0f}s")
                        time.sleep(rest)

        finally:
            browser.close()

    print(f"\n=== Session Complete ===")
    print(f"Downloaded: {total_downloaded} files")
    print(f"Total in archive: {progress['total_files']} files")
    print(f"\nNext: python backend/train/extract_gp_fingering.py \"{DOWNLOAD_DIR}\"")


if __name__ == "__main__":
    main()

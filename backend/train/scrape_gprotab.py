"""
scrape_gprotab.py — GProTab.netからGuitarProファイルを収集
================================================================
アーティストA-Zをクロールし、GPファイルをダウンロード。
ダウンロード後、extract_gp_fingering.py で選好マップに自動追加。

URL構造:
  アーティスト一覧: https://gprotab.net/en/artists/a
  タブページ: https://gprotab.net/en/tabs/artist/song
  ダウンロード: https://gprotab.net/en/tabs/artist/song?download

Usage:
    python backend/train/scrape_gprotab.py --letters a b c
    python backend/train/scrape_gprotab.py --all
    python backend/train/scrape_gprotab.py --artist "metallica"
"""
import os, sys, time, random, re, json
import urllib.request
import urllib.error
from html.parser import HTMLParser
from collections import defaultdict

DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "..", "gprotab_downloads")
BASE_URL = "https://gprotab.net"
DELAY_MIN = 2.0  # 秒 — サーバーに優しく
DELAY_MAX = 5.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class LinkParser(HTMLParser):
    """HTMLからリンクを抽出する簡易パーサー"""
    def __init__(self):
        super().__init__()
        self.links = []
        self._current_href = None
        self._current_text = ""
        self._in_a = False

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            self._current_href = attrs_dict.get("href", "")
            self._current_text = ""
            self._in_a = True

    def handle_data(self, data):
        if self._in_a:
            self._current_text += data.strip()

    def handle_endtag(self, tag):
        if tag == "a" and self._in_a:
            self.links.append((self._current_href, self._current_text))
            self._in_a = False


def fetch_page(url):
    """ページ取得（レート制限付き）"""
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {url}")
        return None
    except Exception as e:
        print(f"  Error: {e}: {url}")
        return None


def get_artists_for_letter(letter):
    """指定文字のアーティスト一覧を取得"""
    url = f"{BASE_URL}/en/artists/{letter}"
    html = fetch_page(url)
    if not html:
        return []

    parser = LinkParser()
    parser.feed(html)

    artists = []
    for href, text in parser.links:
        if href and "/en/tabs/" in href and text:
            artists.append((href, text))

    return artists


def get_tabs_for_artist(artist_url):
    """アーティストページからタブ一覧を取得"""
    if not artist_url.startswith("http"):
        artist_url = BASE_URL + artist_url

    html = fetch_page(artist_url)
    if not html:
        return []

    parser = LinkParser()
    parser.feed(html)

    tabs = []
    for href, text in parser.links:
        if href and "/en/tabs/" in href and text:
            # アーティスト自身のリンクは除外
            parts = href.strip("/").split("/")
            if len(parts) >= 4:  # /en/tabs/artist/song
                tabs.append((href, text))

    return tabs


def download_tab(tab_url, artist_name):
    """タブのGPファイルをダウンロード"""
    if not tab_url.startswith("http"):
        tab_url = BASE_URL + tab_url

    download_url = tab_url + ("&" if "?" in tab_url else "?") + "download"

    # ファイル名を生成
    parts = tab_url.strip("/").split("/")
    if len(parts) >= 2:
        song_name = parts[-1]
    else:
        song_name = "unknown"

    safe_artist = re.sub(r'[^\w\-]', '_', artist_name)
    safe_song = re.sub(r'[^\w\-]', '_', song_name)

    # 保存先
    artist_dir = os.path.join(DOWNLOAD_DIR, safe_artist)
    os.makedirs(artist_dir, exist_ok=True)

    # 既にダウンロード済みか確認
    existing = [f for f in os.listdir(artist_dir) if f.startswith(safe_song)]
    if existing:
        return None  # スキップ

    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    req = urllib.request.Request(download_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()

            if len(data) < 100:
                return None  # HTML（エラーページ）

            # 拡張子を推定
            if b"FICHIER GUITAR PRO" in data[:30] or data[:4] == b"FICH":
                ext = ".gp5"
            elif data[:2] == b"PK":
                ext = ".gpx"
            else:
                ext = ".gp5"

            filepath = os.path.join(artist_dir, f"{safe_song}{ext}")
            with open(filepath, "wb") as f:
                f.write(data)

            return filepath

    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"  Rate limited! Waiting 30s...")
            time.sleep(30)
        return None
    except Exception:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GProTab.net GP file collector")
    parser.add_argument("--letters", nargs="*", help="Letters to crawl (e.g., a b c)")
    parser.add_argument("--all", action="store_true", help="Crawl all letters")
    parser.add_argument("--artist", type=str, help="Specific artist name to search")
    parser.add_argument("--max-artists", type=int, default=10, help="Max artists per letter")
    parser.add_argument("--max-tabs", type=int, default=50, help="Max tabs per artist")
    args = parser.parse_args()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if args.all:
        letters = list("abcdefghijklmnopqrstuvwxyz") + ["0-9"]
    elif args.letters:
        letters = args.letters
    else:
        letters = ["a"]  # デフォルト

    total_downloaded = 0
    progress_file = os.path.join(DOWNLOAD_DIR, "progress.json")

    # 進捗を読み込み
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
    else:
        progress = {"completed_artists": [], "total_files": 0}

    for letter in letters:
        print(f"\n=== Letter: {letter.upper()} ===")
        artists = get_artists_for_letter(letter)
        print(f"  Artists found: {len(artists)}")

        for i, (artist_url, artist_name) in enumerate(artists[:args.max_artists]):
            if artist_name in progress["completed_artists"]:
                print(f"  [{i+1}] {artist_name} — skipped (already done)")
                continue

            print(f"  [{i+1}] {artist_name}...")
            tabs = get_tabs_for_artist(artist_url)
            print(f"      Tabs: {len(tabs)}")

            artist_downloads = 0
            for tab_url, tab_name in tabs[:args.max_tabs]:
                filepath = download_tab(tab_url, artist_name)
                if filepath:
                    print(f"      OK: {tab_name} -> {os.path.basename(filepath)}")
                    artist_downloads += 1
                    total_downloaded += 1

            progress["completed_artists"].append(artist_name)
            progress["total_files"] += artist_downloads

            # 進捗を保存
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)

    print(f"\n=== Complete ===")
    print(f"Total downloaded: {total_downloaded}")
    print(f"Saved to: {DOWNLOAD_DIR}")
    print(f"\n次のステップ:")
    print(f"  python backend/train/extract_gp_fingering.py \"{DOWNLOAD_DIR}\"")


if __name__ == "__main__":
    main()

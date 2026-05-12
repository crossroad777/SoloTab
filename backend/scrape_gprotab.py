"""
GProTab GP5 Scraper - Fingerstyle GP5 file collection
=============================================================
Approach: Browse A-Z artist index -> artist pages -> song pages -> download GP files
Site: https://www.gprotab.net
Purpose: Research/Academic use only (Private Study).
"""

import os
import sys
import time
import json
import re
import requests
from pathlib import Path
from urllib.parse import urljoin

# --- Config ---
BASE_URL = "https://www.gprotab.net"
OUTPUT_DIR = Path(r"D:\Music\nextchord-solotab\datasets\gprotab_downloads")
METADATA_FILE = OUTPUT_DIR / "metadata.json"
DELAY = 2.0  # seconds between requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def safe_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip()[:100]


def load_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"downloaded": [], "failed": [], "total": 0, "artists_done": []}


def save_metadata(meta):
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def fetch_html(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  [ERR] {url}: {e}")
        return ""


def get_artists_for_letter(letter):
    """Get all artist links from a letter page"""
    url = f"{BASE_URL}/en/artists/{letter}"
    html = fetch_html(url)
    if not html:
        return []
    # Pattern: /en/tabs/artist-slug (artist pages, not song pages)
    pattern = r'href="(/en/tabs/[^/"]+)"'
    matches = re.findall(pattern, html)
    # Deduplicate
    return list(set(matches))


def get_songs_for_artist(artist_path):
    """Get all song links from an artist page"""
    url = urljoin(BASE_URL, artist_path)
    html = fetch_html(url)
    if not html:
        return []
    # Songs have pattern: /en/tabs/artist-slug/song-slug
    artist_slug = artist_path.strip('/').split('/')[-1]
    pattern = rf'href="(/en/tabs/{re.escape(artist_slug)}/[^/"]+)"'
    matches = re.findall(pattern, html)
    return list(set(matches))


def find_and_download(song_path, meta):
    """Visit song page, find download link, download GP file"""
    song_url = urljoin(BASE_URL, song_path)
    
    # Check if already done
    done_urls = set(d.get('url', '') for d in meta['downloaded'])
    if song_url in done_urls or song_url in meta['failed']:
        return False
    
    html = fetch_html(song_url)
    if not html:
        meta['failed'].append(song_url)
        return False
    
    # Find download link - look for GP file links
    # Common patterns on GProTab:
    # 1. Direct file link: *.gp5, *.gp4, *.gpx etc
    # 2. Download button/link
    dl_patterns = [
        r'href="([^"]*\.(gp[345x]|gpx?))"',
        r'href="([^"]*download[^"]*)"',
        r'href="(/files/[^"]+)"',
        r'data-url="([^"]*\.(gp[345x]|gpx?))"',
        r"src=['\"]([^'\"]*\.(gp[345x]|gpx?))['\"]",
    ]
    
    dl_link = None
    for pattern in dl_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            dl_link = match.group(1)
            break
    
    if not dl_link:
        # Try looking for any button/link with "download" text
        download_section = re.search(r'class="[^"]*download[^"]*"[^>]*href="([^"]+)"', html, re.IGNORECASE)
        if download_section:
            dl_link = download_section.group(1)
    
    if not dl_link:
        meta['failed'].append(song_url)
        return False
    
    if not dl_link.startswith('http'):
        dl_link = urljoin(BASE_URL, dl_link)
    
    # Extract artist/title from path
    parts = song_path.strip('/').split('/')
    artist = safe_filename(parts[2]) if len(parts) > 2 else "unknown"
    title = safe_filename(parts[3]) if len(parts) > 3 else safe_filename(parts[-1])
    
    # Determine extension
    ext = '.gp5'
    for e in ['.gp5', '.gp4', '.gp3', '.gpx', '.gp']:
        if e in dl_link.lower():
            ext = e
            break
    
    save_path = OUTPUT_DIR / artist / f"{title}{ext}"
    
    # Download
    time.sleep(DELAY)
    try:
        resp = requests.get(dl_link, headers=HEADERS, timeout=60, allow_redirects=True)
        if resp.status_code != 200:
            meta['failed'].append(song_url)
            return False
        
        content_type = resp.headers.get('Content-Type', '')
        if 'html' in content_type.lower() and len(resp.content) < 5000:
            meta['failed'].append(song_url)
            return False
        
        if len(resp.content) < 100:
            meta['failed'].append(song_url)
            return False
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        
        meta['downloaded'].append({
            'url': song_url,
            'dl_url': dl_link,
            'artist': artist,
            'title': title,
            'path': str(save_path),
            'size': len(resp.content),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        })
        meta['total'] += 1
        return True
    except Exception as e:
        print(f"    [DL ERR] {e}")
        meta['failed'].append(song_url)
        return False


def main():
    print("=" * 60)
    print("  GProTab GP5 Scraper")
    print("=" * 60)
    
    meta = load_metadata()
    print(f"  Previously downloaded: {meta['total']} files")
    print(f"  Output: {OUTPUT_DIR}")
    
    letters = list('abcdefghijklmnopqrstuvwxyz') + ['09']
    done_artists = set(meta.get('artists_done', []))
    
    for letter in letters:
        print(f"\n--- Letter: {letter.upper()} ---")
        time.sleep(DELAY)
        
        artists = get_artists_for_letter(letter)
        print(f"  Found {len(artists)} artists")
        
        for artist_path in sorted(artists):
            artist_slug = artist_path.strip('/').split('/')[-1]
            
            if artist_slug in done_artists:
                continue
            
            time.sleep(DELAY)
            songs = get_songs_for_artist(artist_path)
            
            if not songs:
                done_artists.add(artist_slug)
                meta['artists_done'] = list(done_artists)
                continue
            
            dl_count = 0
            for song_path in songs:
                time.sleep(DELAY)
                if find_and_download(song_path, meta):
                    dl_count += 1
                    title = song_path.strip('/').split('/')[-1]
                    print(f"    [{meta['total']}] OK: {artist_slug}/{title}")
                
                # Save every 10 downloads
                if meta['total'] % 10 == 0 and meta['total'] > 0:
                    save_metadata(meta)
            
            done_artists.add(artist_slug)
            meta['artists_done'] = list(done_artists)
            
            if dl_count > 0:
                print(f"  {artist_slug}: {dl_count}/{len(songs)} downloaded")
            
            save_metadata(meta)
    
    save_metadata(meta)
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {meta['total']} total files")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  [INTERRUPTED]")
        sys.exit(0)

"""Refresh the public directory or reproduce an import from the factual snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crbot.deckbuilder_import import SOURCE_URL, parse_database, parse_details, merge_directory
from crbot.knowledge import KnowledgeBase
from crbot.cards import CardCatalog


def download(url):
    with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0 (RoyalLab public card importer)'}), timeout=30) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--icons', action='store_true')
    args = parser.parse_args()
    snapshot_path = Path('data/deckbuilder_cards.json')
    if args.refresh:
        page = download(SOURCE_URL)
        urls = re.findall(r'<script[^>]+src="([^"]+)"', page.decode('utf8'))
        # Discover only scripts explicitly linked by the source page.
        found = []
        for relative in urls:
            if not relative.startswith('/_next/static/chunks/'):
                continue
            url = urljoin(SOURCE_URL, relative)
            script = download(url)
            if b'FINAL_CARDS_DATABASE:()' in script:
                found.append((url, script))
        if len(found) != 1:
            raise ValueError('Source database layout changed; no files updated')
        url, script = found[0]
        snapshot = dict(schema_version=1, source=dict(url=SOURCE_URL, asset_url=url,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            page_sha256=hashlib.sha256(page).hexdigest(), asset_sha256=hashlib.sha256(script).hexdigest(),
            status='public_detail_stats_unvalidated_balance', fields=['name', 'cost', 'rarity', 'type', 'arena', 'icon', 'tags', 'evolution', 'champion', 'battle_stats']),
            cards=parse_database(script.decode('utf8')))
        detail_page = download(SOURCE_URL + '/knight')
        detail_urls = re.findall(r'<script[^>]+src="([^"]+)"', detail_page.decode('utf8'))
        detail_url = next(u for u in detail_urls if '/%5Bid%5D/page-' in u)
        detail_url = urljoin(SOURCE_URL, detail_url)
        detail_script = download(detail_url)
        details = parse_details(detail_script.decode('utf8'))
        snapshot['source'].update(detail_asset_url=detail_url,
            detail_asset_sha256=hashlib.sha256(detail_script).hexdigest())
        for row in snapshot['cards']:
            row['battle_stats'] = details[row['id']]
    else:
        snapshot = json.loads(snapshot_path.read_text(encoding='utf8'))
    catalog, kb = merge_directory(json.loads(Path('data/cards.json').read_text(encoding='utf8')),
                                  json.loads(Path('data/battle_knowledge.json').read_text(encoding='utf8')), snapshot)
    if args.icons:
        from PIL import Image
        from io import BytesIO
        for card in catalog['cards']:
            if card.get('icon_path') or not card.get('deckbuilder_current'):
                continue
            content = download(card['icon_url'])
            Image.open(BytesIO(content)).verify()
            path = Path('data/card_icons') / (card['id'] + '.png')
            path.write_bytes(content)
            card['icon_path'] = 'card_icons/' + path.name
    KnowledgeBase(kb)
    for path, payload in [(snapshot_path, snapshot), (Path('data/cards.json'), catalog), (Path('data/battle_knowledge.json'), kb)]:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2 if 'battle_' not in path.name else None) + '\n', encoding='utf8')
        if path.name == 'cards.json':
            CardCatalog.load(temporary)
        temporary.replace(path)
    print(json.dumps(KnowledgeBase(kb).audit(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

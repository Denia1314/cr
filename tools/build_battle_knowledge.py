"""Import a pinned public RoyaleAPI snapshot without replacing legacy card definitions.

Run from the repository root. Source revision is mandatory; current-balance validation
is a separate process. Downloaded public numeric data is not battle data or a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crbot.knowledge import KnowledgeBase, build_knowledge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True, help="40-character RoyaleAPI/cr-api-data git commit")
    parser.add_argument("--catalog", default="data/cards.json")
    parser.add_argument("--output", default="data/battle_knowledge.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision 必须是完整 commit SHA")
    base = f"https://raw.githubusercontent.com/RoyaleAPI/cr-api-data/{args.revision}/docs/json/"
    source_bytes = {}
    for name in ("cards_stats", "rarities"):
        req = Request(base + name + ".json", headers={"User-Agent": "RoyalLab/3"})
        with urlopen(req, timeout=30) as response:
            source_bytes[name] = response.read()
    source = {"repository": "https://github.com/RoyaleAPI/cr-api-data", "revision": args.revision,
              "urls": [base + name + ".json" for name in source_bytes],
              "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in source_bytes.items()},
              "retrieved_at": datetime.now(timezone.utc).isoformat(),
              "balance_version": "unknown", "status": "historical_unvalidated"}
    payload = build_knowledge(json.loads(Path(args.catalog).read_text(encoding="utf-8")),
                              json.loads(source_bytes["cards_stats"]), json.loads(source_bytes["rarities"]), source)
    kb = KnowledgeBase(payload)
    audit = kb.audit()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Create one inspectable effect record per catalog card from the imported public data."""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crbot.knowledge import KnowledgeBase
from crbot.card_effects import mechanism_profile


def main():
    kb=KnowledgeBase.load(Path('data/battle_knowledge.json'))
    payload=dict(schema_version=1,knowledge_version=kb.version,
                 cards={cid:mechanism_profile(kb,cid) for cid in kb.cards})
    Path('data/card_effects.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf8')
    print('effect records:',len(payload['cards']))


if __name__ == '__main__':
    main()

import copy
import json
from pathlib import Path
import unittest

from crbot.cards import CardCatalog
from crbot.deckbuilder_import import merge_directory, parse_database, parse_details
from crbot.knowledge import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / 'data' / name).read_text(encoding='utf8'))


class DeckbuilderImportTests(unittest.TestCase):
    def test_explicit_levels_match_user_screenshot(self):
        kb = KnowledgeBase(read('battle_knowledge.json'))
        low = kb.roster('knight', 3)[0][0]
        current = kb.roster('knight', 11)[0][0]
        self.assertEqual((low.hp, low.damage), (834, 95))
        self.assertEqual((current.hp, current.damage, current.period), (1766, 202, 1.2))
        with self.assertRaises(ValueError):
            kb.roster('knight', 2)

    def test_full_directory_and_new_icons_are_present(self):
        snapshot = read('deckbuilder_cards.json')
        catalog = CardCatalog.load(ROOT / 'data/cards.json')
        self.assertEqual(len(snapshot['cards']), 121)
        self.assertEqual(sum(len(r['battle_stats']['levelStats']) for r in snapshot['cards']), 1083)
        self.assertEqual(len({r['id'] for r in snapshot['cards']}), 121)
        for row in snapshot['cards']:
            card = catalog.get(row['id'])
            self.assertIsNotNone(card)
            self.assertTrue(catalog.local_icon_path(card).is_file())

    def test_import_is_idempotent_and_preserves_legacy_strategy(self):
        catalog, kb = read('cards.json'), read('battle_knowledge.json')
        catalog['cards'][0]['strategy_verified'] = True
        catalog['cards'][0]['counters'] = ['test_curated_counter']
        source_before = copy.deepcopy(kb['source'])
        units_before = copy.deepcopy(kb['units']['Knight'])
        first = merge_directory(catalog, kb, read('deckbuilder_cards.json'))
        second = merge_directory(*first, read('deckbuilder_cards.json'))
        self.assertEqual(first, second)
        self.assertTrue(first[0]['cards'][0]['strategy_verified'])
        self.assertEqual(first[0]['cards'][0]['counters'], ['test_curated_counter'])
        self.assertEqual(first[1]['source'], source_before)
        self.assertEqual(first[1]['units']['Knight'], units_before)
        self.assertEqual({c['id'] for c in catalog['cards']}, {c['id'] for c in first[0]['cards']})

    def test_data_is_not_mechanism_acceptance_and_mirror_is_not_free(self):
        kb = KnowledgeBase(read('battle_knowledge.json'))
        self.assertTrue(kb.cards['goblinstein']['detail_stats']['levelStats'])
        self.assertEqual(kb.roster('goblinstein'), [])
        self.assertIsNone(kb.cards['mirror']['elixir'])
        self.assertFalse(kb.audit()['validated_current_balance'])
        self.assertFalse(kb.audit()['battle_acceptance'])

    def test_literal_decoder_rejects_code_and_keeps_special_fields(self):
        script = 'let m={knight:{cardId:"knight",hitSpeed:.5,levelStats:[{level:11,hitpoints:1e3,shieldHitpoints:99}]}};'
        row = parse_details(script)['knight']
        self.assertEqual(row['hitSpeed'], .5)
        self.assertEqual(row['levelStats'][0]['hitpoints'], 1000)
        self.assertEqual(row['levelStats'][0]['shieldHitpoints'], 99)
        with self.assertRaises(ValueError):
            parse_details('let m={knight:evil()};')
        with self.assertRaises(ValueError):
            parse_database('s("knight","Knight",3)')

    def test_mixed_roster_and_tower_damage_are_not_conflated(self):
        kb = KnowledgeBase(read('battle_knowledge.json'))
        self.assertGreater(len(kb.roster('goblin_gang')), 1)
        spell = kb.spell('fireball', 11)
        source = next(r for r in kb.cards['fireball']['detail_stats']['levelStats'] if r['level'] == 11)
        self.assertAlmostEqual(spell['damage'] * spell['tower_multiplier'], source['crownTowerDamage'])
        self.assertEqual(kb.spell('lightning')['pattern'], 'highest_hp')


if __name__ == '__main__':
    unittest.main()

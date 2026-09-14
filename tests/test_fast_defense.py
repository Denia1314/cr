from dataclasses import replace
import unittest
from PIL import Image, ImageDraw
from crbot.unit_health import read_unit_health
from crbot.battle_world import BattleWorld
from crbot.predictive_planner import PredictivePlanner
from tests.test_prediction import knowledge, world


class FastDefenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = knowledge()

    def test_health_bar_requires_complete_unambiguous_frame(self):
        image = Image.new("RGB", (400, 800), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((90, 190, 131, 195), fill=(20,20,20))
        draw.rectangle((91, 191, 110, 194), fill=(210,40,40))
        bbox = (.20,.25,.35,.35)
        self.assertEqual(read_unit_health(image, bbox)["hp_fraction"], .5)
        self.assertEqual(read_unit_health(Image.new("RGB", image.size, "white"), bbox), {})
        draw.rectangle((90,190,131,190), fill="white")
        self.assertEqual(read_unit_health(image, bbox), {})

    def test_health_updates_allow_healing_and_expire(self):
        w = BattleWorld()
        d = dict(card_id="knight", side=-1, x=.28,y=.5,confidence=.9,hp_fraction=.1)
        def update(now, detection):
            return w.update([detection], now=now, elapsed=now, elixir=7, hand=[],costs={},seconds_per_elixir=2.8)
        self.assertEqual(update(1,d).tracks[0].hp_fraction,.1)
        self.assertEqual(update(1.1,{**d,"hp_fraction":.5}).tracks[0].hp_fraction,.5)
        self.assertEqual(update(1.2,{**d,"hp_fraction":float("nan")}).tracks[0].hp_fraction,.5)
        self.assertIsNone(update(2,{**d,"hp_fraction":None}).tracks[0].hp_fraction)

    def test_fast_search_compares_every_usable_card_without_refinement(self):
        p = PredictivePlanner(self.kb, {"fast_defense":True,"budget_ms":2000})
        result = p.plan(world())
        self.assertEqual(result.completed_depth,1)
        self.assertFalse(result.budget_exhausted)
        self.assertTrue(all(e["status"] == "complete" for e in result.hand_evaluations))
        self.assertEqual(len(result.hand_evaluations),4)

    def test_tower_can_finish_low_health_enemy_without_extra_card(self):
        w = world()
        w = replace(w, tracks=(replace(w.tracks[0],card_id="knight",y=.65,hp_fraction=.01),))
        result = PredictivePlanner(self.kb,{"fast_defense":True,"budget_ms":2000}).plan(w)
        self.assertEqual(result.status,"wait")
        self.assertIsNone(result.action.card_id)

    def test_healthy_push_still_receives_defense(self):
        result = PredictivePlanner(self.kb,{"fast_defense":True,"budget_ms":2000}).plan(world())
        self.assertEqual(result.status,"ready")

    def test_zero_health_not_spawned_into_simulation(self):
        w = world()
        w = replace(w,tracks=(replace(w.tracks[0],hp_fraction=0),))
        initial = PredictivePlanner(self.kb,{}).initial(w,7,1)
        self.assertFalse(any(e.side==-1 and not e.tower for e in initial.entities))

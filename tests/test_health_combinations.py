import time
import unittest
from dataclasses import replace
from PIL import Image,ImageDraw
from crbot.unit_health import read_unit_health,detect_unit_health_bars
from crbot.battle_world import Track
from crbot.battle_simulation import SimAction
from crbot.predictive_planner import PredictivePlanner
from crbot.parallel_combat import CombatPool
from tests.test_prediction import world,knowledge


class HealthCombinationTests(unittest.TestCase):
    def test_both_teams_visible_bars_and_unknown_absence(self):
        image=Image.new('RGB',(400,800),'white');draw=ImageDraw.Draw(image)
        for x,color in ((80,(230,40,40)),(260,(40,100,230))):
            draw.rectangle((x,390,x+41,395),fill=(20,20,20))
            draw.rectangle((x+1,391,x+20,394),fill=color)
        bars=detect_unit_health_bars(image)
        self.assertEqual({b['side'] for b in bars},{-1,1})
        self.assertTrue(all(b['hp_fraction']==.5 for b in bars))
        self.assertEqual(detect_unit_health_bars(Image.new('RGB',image.size,'white')),[])
        self.assertEqual(detect_unit_health_bars(image,[(.25,.49),(.70,.49)]),[])

    def test_pink_and_cyan_gradient_bars_are_read_without_swapping_sides(self):
        for side,color,background in ((-1,(250,190,235),(100,40,70)),(1,(80,220,250),(30,70,100))):
            image=Image.new('RGB',(400,800),'white');draw=ImageDraw.Draw(image)
            draw.rectangle((90,390,131,395),fill=background)
            draw.rectangle((91,391,110,394),fill=color)
            box=(.2,.5,.35,.6)
            self.assertEqual(read_unit_health(image,box,side)['hp_fraction'],.5)
            self.assertEqual(read_unit_health(image,box,-side),{})
            self.assertEqual(read_unit_health(image,(.22,.5,.5,.6),side),{})
            draw.rectangle((90,390,131,390),fill='white')
            self.assertEqual(read_unit_health(image,box,side),{})

    def test_observed_ally_health_changes_surviving_material_and_stale_health_is_not_reused(self):
        p=PredictivePlanner(knowledge(),{})
        ally=Track(1,'knight',1,.28,.65,99,100,.9,hp_fraction=.1,hp_observed_at=100,hp_confidence=.85)
        low=p.initial(world(tracks=(ally,)),7,1)
        high=p.initial(world(tracks=(replace(ally,hp_fraction=.9),)),7,1)
        self.assertLess(p.sim.evaluate(low)[1]['material'],p.sim.evaluate(high)[1]['material'])
        stale=p.initial(world(tracks=(replace(ally,hp_observed_at=98),)),7,1)
        self.assertNotEqual(next(e for e in stale.entities if not e.tower).hp,next(e for e in low.entities if not e.tower).hp)

    def test_two_card_defense_reduces_tower_loss_with_recomputed_followup_positions(self):
        p=PredictivePlanner(knowledge(),dict(fast_defense=True,unified_tactics=True));p.clock=lambda:0.
        snapshot=world(elixir=10,hand=((0,'knight'),(1,'musketeer'),(2,'fireball')),
            tracks=(Track(1,'giant',-1,.28,.62,99,100,.9,hp_fraction=1),Track(2,'musketeer',-1,.28,.53,99,100,.9,hp_fraction=1)))
        root=SimAction('musketeer',1,.3,.73);scenarios=[(7,1,False)]
        single=p.evaluate_root(snapshot,root,scenarios,8,'defend',100)
        pair=p.evaluate_combination(snapshot,root,scenarios,8,'defend',100)
        self.assertLess(pair['branches'][0]['own_tower_damage'],single['branches'][0]['own_tower_damage'])
        self.assertEqual(pair['branches'][0]['followup_action']['card_id'],'fireball')
        self.assertGreater(pair['branches'][0]['followup_candidates'],3)
        self.assertEqual(pair['planned_cost'],8)

    def test_parallel_combinations_match_serial_including_followup_locations(self):
        kb=knowledge();config=dict(fast_defense=True,unified_tactics=True,combat_workers=2)
        p=PredictivePlanner(kb,config);pool=CombatPool(kb.payload,config)
        try:
            snapshot=world();scenarios=[(0,.65,False),(7,1,False)]
            roots=[SimAction(),SimAction('knight',0,.3,.65)]
            serial=[p.evaluate_combination(snapshot,r,scenarios,8,'defend',time.perf_counter()+10) for r in roots]
            parallel=list(pool.rows(snapshot,roots,scenarios,8,'defend',time.perf_counter()+10,kind='combo'))
            self.assertEqual(serial,parallel)
            self.assertEqual(pool.completed_combinations,2)
        finally:pool.close()


if __name__=='__main__':unittest.main()

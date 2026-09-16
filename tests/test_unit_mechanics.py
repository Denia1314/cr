from dataclasses import replace
import unittest

from crbot.battle_simulation import Simulator, SimState, SimAction
from crbot.unit_mechanics import deployment_pulses, control, mechanism_status, placement_special_bonus
from crbot.grid_world import target_for
from tests.test_prediction import knowledge


class UnitMechanicTests(unittest.TestCase):
    def setUp(self):
        self.kb=knowledge();self.sim=Simulator(self.kb,step=.05)

    def target(self,s,x=5,y=20,side=-1,air=False,shield=0):
        spec=replace(self.kb.unit('Knight'),hp=10000,damage=0,speed=0,air=air,shield=shield,radius=.3)
        return self.sim.add(s,spec,side,x,y)

    def cast(self,s,cid,x=5,y=20):
        s.elixir[1]=10;s.hands[1]=[(0,cid)]
        sx,sy=self.sim.screen(x,y)
        self.assertTrue(self.sim.apply(s,SimAction(cid,0,sx,sy),1))

    def test_mega_knight_spawn_is_delayed_once_and_ground_only(self):
        s=SimState();enemy=self.target(s);air=self.target(s,air=True);ally=self.target(s,side=1)
        far=self.target(s,x=10)
        self.cast(s,'mega_knight')
        expected=deployment_pulses(self.kb,'mega_knight',11)[0]['damage']
        self.sim.advance(s,.9);self.assertEqual(enemy.hp,10000)
        self.sim.advance(s,.2);self.assertAlmostEqual(10000-enemy.hp,expected)
        self.assertEqual([air.hp,ally.hp,far.hp],[10000]*3)
        self.sim.advance(s,.3);self.assertAlmostEqual(10000-enemy.hp,expected)

    def test_observed_existing_units_never_replay_spawn_damage(self):
        for cid in ['mega_knight','electro_wizard','ice_wizard']:
            with self.subTest(cid=cid):
                s=SimState();spec=self.kb.roster(cid)[0][0]
                self.sim.add(s,spec,1,5,20)
                self.assertEqual(s.effects,[])

    def test_electro_and_ice_wizard_spawn_controls_use_database(self):
        for cid,field in [('electro_wizard','stunned_until'),('ice_wizard','slow_until')]:
            with self.subTest(cid=cid):
                s=SimState();enemy=self.target(s,air=True)
                self.cast(s,cid);self.sim.advance(s,1.05)
                self.assertLess(enemy.hp,10000)
                self.assertGreater(getattr(enemy,field),s.time)
                if cid=='ice_wizard':
                    self.assertAlmostEqual(enemy.slow,.65)
                    self.assertAlmostEqual(enemy.attack_slow,.65)

    def jumper(self,s,cid='mega_knight',distance=4):
        spec=replace(self.kb.roster(cid)[0][0],damage=0,speed=0)
        entity=self.sim.add(s,spec,1,5,20)
        victim=self.target(s,x=5+distance+spec.radius+.3)
        return entity,victim

    def test_jump_has_windup_flight_and_one_landing_hit(self):
        s=SimState();jumper,victim=self.jumper(s)
        self.sim.advance(s,.5)
        self.assertEqual(jumper.dash_phase,'windup');self.assertEqual(victim.hp,10000)
        self.sim.advance(s,.6)
        self.assertEqual(jumper.dash_phase,'flight');self.assertEqual(victim.hp,10000)
        clone=s.clone();clone.entities[0].dash_x=0
        self.assertNotEqual(jumper.dash_x,clone.entities[0].dash_x)
        self.sim.advance(s,1.2)
        self.assertAlmostEqual(10000-victim.hp,jumper.spec.dash_damage)
        self.sim.advance(s,.5)
        self.assertAlmostEqual(10000-victim.hp,jumper.spec.dash_damage)

    def test_near_and_out_of_range_units_do_not_jump(self):
        for distance in [1.,6.]:
            s=SimState();jumper,victim=self.jumper(s,distance=distance)
            self.sim.advance(s,3)
            self.assertEqual(jumper.dash_phase,'');self.assertEqual(victim.hp,10000)

    def test_stun_interrupts_windup_and_dead_target_cancels(self):
        s=SimState();jumper,victim=self.jumper(s)
        self.sim.advance(s,.3);control(s,jumper,stun=2)
        self.assertEqual(jumper.dash_phase,'')
        self.sim.advance(s,1);self.assertEqual(victim.hp,10000)
        s=SimState();jumper,victim=self.jumper(s)
        self.sim.advance(s,.3);victim.hp=0;self.sim.advance(s,.1)
        self.assertEqual(jumper.dash_phase,'')

    def test_bandit_immunity_only_during_flight(self):
        s=SimState();bandit,victim=self.jumper(s,'bandit')
        self.sim.advance(s,.3);before=bandit.hp
        self.assertTrue(self.sim.hit(s,bandit,10));self.assertEqual(bandit.hp,before-10)
        self.sim.advance(s,.6);self.assertEqual(bandit.dash_phase,'flight')
        before=bandit.hp;self.assertFalse(self.sim.hit(s,bandit,10));self.assertEqual(bandit.hp,before)
        self.sim.advance(s,2);self.assertTrue(self.sim.hit(s,bandit,10))

    def test_reflection_requires_nearby_attack_source_not_a_spell(self):
        s=SimState();giant=self.sim.add(s,self.kb.roster('electro_giant')[0][0],-1,5,20)
        attacker=self.target(s,x=6,side=1)
        self.sim.hit(s,giant,10,attacker)
        self.assertEqual(10000-attacker.hp,giant.spec.reflected_damage)
        self.assertGreater(attacker.stunned_until,s.time)
        before=attacker.hp;self.sim.hit(s,giant,10)
        self.assertEqual(attacker.hp,before)
        attacker.x=15;self.sim.hit(s,giant,10,attacker)
        self.assertEqual(attacker.hp,before)

    def test_dual_bolts_split_or_hit_lone_target_twice(self):
        for count in [1,2]:
            s=SimState();spec=replace(self.kb.roster('electro_wizard')[0][0],first_hit=0,speed=0)
            self.sim.add(s,spec,1,5,20)
            targets=[self.target(s,x=6+i) for i in range(count)]
            self.sim.advance(s,.15)
            self.assertAlmostEqual(sum(10000-e.hp for e in targets),spec.damage*2)
            if count==2:self.assertEqual(targets[0].hp,targets[1].hp)

    def test_minimum_range_and_pushback_immunity(self):
        s=SimState();spec=replace(self.kb.unit('Knight'),minimum_range=3,reach=10)
        mortar=self.sim.add(s,spec,1,5,20)
        near=self.target(s,x=6);far=self.target(s,x=10)
        self.assertIs(target_for(self.sim,s,mortar),far)
        mega=self.sim.add(s,self.kb.roster('mega_knight')[0][0],1,5,20)
        control(s,mega,pushback=2,x=4,y=20)
        self.assertEqual(mega.x,5)

    def test_level_values_and_coverage_remain_explicit(self):
        self.assertGreater(self.kb.roster('mega_knight',12)[0][0].dash_damage,self.kb.roster('mega_knight',11)[0][0].dash_damage)
        status=mechanism_status(self.kb,'mega_knight')
        self.assertIn('deployment_damage',status['implemented'])
        self.assertIn('jump_or_dash',status['implemented'])
        self.assertIn('evolution_or_hero_variant',status['pending'])
        self.assertFalse(status['validated_current_balance'])
        self.assertIn('multiple_projectiles',mechanism_status(self.kb,'hunter')['pending'])

    def test_shortlist_accounts_for_spawn_damage_before_combat_search(self):
        s=SimState();victim=self.target(s)
        roster=self.kb.roster('mega_knight')
        effects=deployment_pulses(self.kb,'mega_knight',11)
        projected=[(victim,(5,20))]
        with_spawn=placement_special_bonus((5,20),roster,projected,effects)
        without=placement_special_bonus((5,20),roster,projected,[])
        self.assertGreater(with_spawn,without)
        self.assertEqual(placement_special_bonus((15,20),roster,projected,effects),0)

    def test_chain_hits_three_distinct_targets_within_hop_range(self):
        s=SimState();spec=replace(self.kb.roster('electro_dragon')[0][0],first_hit=0,speed=0,period=10,projectile_speed=1000)
        self.assertEqual(spec.chain_count,3)
        self.sim.add(s,spec,1,5,20)
        victims=[self.target(s,x=x) for x in [6,8,10,12]]
        self.sim.advance(s,.3)
        self.assertEqual(sum(v.hp<10000 for v in victims),3)
        self.assertAlmostEqual(sum(10000-v.hp for v in victims),3*spec.damage)

    def test_missing_death_bomb_definition_is_never_claimed_as_implemented(self):
        status=mechanism_status(self.kb,'giant_skeleton')
        self.assertIn('death_spawn_missing_definition',status['pending'])
        self.assertNotIn('death_spawn',status['implemented'])


if __name__=='__main__':unittest.main()

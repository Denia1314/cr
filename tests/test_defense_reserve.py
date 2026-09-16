import unittest
from unittest.mock import patch
from crbot.battle_simulation import SimAction
from crbot.predictive_planner import PredictivePlanner
from crbot.tactical_objective import guard_development_reserve
from tests.test_prediction import knowledge, world
from tests.test_urgent_tower_defense import row


class DefenseReserveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = knowledge()

    def scenario(self, elixir=6):
        return world(elixir=elixir, tracks=(), hand=((0,'knight'),(1,'musketeer'),(2,'ice_golem'),(3,'electro_spirit')))

    def test_crisis_equal_worst_branch_uses_reserve_and_mean_damage(self):
        from crbot.battle_world import Track
        from crbot.tactical_objective import emergency_tower_defense
        p=PredictivePlanner(self.kb,{})
        w=world(elixir=4,hand=((0,'knight'),(1,'musketeer')),
                tracks=(Track(1,'giant',-1,.3,.6,99,100,.9),))
        waiting=row(SimAction(),3051.4)
        weak=row(SimAction('knight',0,.3,.65),3051.4)
        strong=row(SimAction('musketeer',1,.3,.7),3051.4)
        for r,damage in ((waiting,2500),(weak,1900),(strong,1000)):
            r['branches'].append(dict(r['branches'][0],own_tower_damage=damage))
        result,audit=emergency_tower_defense(waiting,[waiting,weak,strong],w,p.sim,p.initial(w,10,1))
        self.assertIs(result,strong)
        self.assertTrue(audit['reserve_override'])
        self.assertFalse(audit['worst_risk_improved'])
        self.assertTrue(audit['mean_risk_improved'])

    def test_crisis_acts_without_proven_mitigation_but_not_below_threshold(self):
        from crbot.battle_world import Track
        from crbot.tactical_objective import emergency_tower_defense
        p=PredictivePlanner(self.kb,{})
        for enemy,elixir,damage,expected in (('giant',3,3051.4,True),('giant',3,100,False),
                                            ('giant',2,3051.4,False),('balloon',3,3051.4,False)):
            with self.subTest(enemy=enemy,elixir=elixir,damage=damage):
                w=world(elixir=elixir,hand=((0,'knight'),),tracks=(Track(1,enemy,-1,.3,.6,99,100,.9),))
                waiting=row(SimAction(),damage);play=row(SimAction('knight',0,.3,.65),damage)
                result,audit=emergency_tower_defense(waiting,[waiting,play],w,p.sim,p.initial(w,10,1))
                self.assertEqual(audit['changed'],expected)
                self.assertIs(result,play if expected else waiting)

    def test_planner_crisis_bypasses_reserve_with_no_proven_reduction(self):
        from crbot.battle_world import Track
        p=PredictivePlanner(self.kb,dict(fast_defense=True,unified_tactics=True,development_reserve_gate=True))
        p.clock=lambda:0
        w=world(elixir=3,hand=((0,'knight'),),tracks=(Track(1,'giant',-1,.3,.6,99,100,.9),))
        roots=[SimAction(),SimAction('knight',0,.3,.65)]
        with patch.object(p,'candidates',return_value=roots), \
             patch.object(p,'evaluate_root',side_effect=lambda w,a,*args:row(a,3051.4)), \
             patch.object(p,'refine_combinations',return_value=[]):
            result=p.plan(w)
        self.assertEqual(result.status,'ready')
        self.assertTrue(result.compute['tower_emergency_defense']['reserve_override'])

    def test_uncertain_balloon_hypothesis_does_not_block_ground_defense(self):
        from crbot.battle_world import Track
        from crbot.tactical_objective import emergency_tower_defense
        p=PredictivePlanner(self.kb,{})
        w=world(elixir=3,hand=((0,'knight'),),tracks=(Track(1,'unknown:left:single',-1,.3,.6,99,100,.9,
                hypotheses=('knight','minions','balloon')),))
        waiting=row(SimAction(),3051.4);play=row(SimAction('knight',0,.3,.65),3051.4)
        chosen,audit=emergency_tower_defense(waiting,[waiting,play],w,p.sim,p.initial(w,10,1))
        self.assertIs(chosen,play)
        self.assertTrue(audit['changed'])

    def test_remaining_four_cost_defender_needs_four_not_three(self):
        waiting = row(SimAction(),0)
        play = row(SimAction('knight',0,.3,.72),0,10)
        choices, status = guard_development_reserve([waiting,play],self.scenario(),self.kb,'develop')
        self.assertEqual(choices,[waiting])
        option = status['options'][0]
        self.assertEqual(option['required'],4)
        self.assertEqual(option['elixir_after'],3)
        self.assertEqual(option['retained_defenders'],['musketeer'])
        self.assertFalse(status['future_draw_assumed'])
        choices, _ = guard_development_reserve([waiting,play],self.scenario(7),self.kb,'develop')
        self.assertIn(play,choices)

    def test_preparation_wait_yields_to_immediate_tower_protection(self):
        for damage, exposure, expected in ((400,0,'ready'), (0,400,'ready'), (0,0,'wait')):
            with self.subTest(damage=damage, exposure=exposure):
                p=PredictivePlanner(self.kb,dict(fast_defense=True,unified_tactics=True,development_reserve_gate=True))
                p.clock=lambda:0
                roots=[SimAction(),SimAction('knight',0,.3,.72)]
                def evaluate(w,a,*args):
                    result=row(a,0 if a.card_id else damage,-100 if a.card_id else 100)
                    result['branches'][0]['imminent_tower_exposure']=0 if a.card_id else exposure
                    return result
                # A high scoring conditional WAIT must not hide the damage of
                # actually doing nothing. No future follow-up is yet executed.
                combo=row(SimAction(),0,1000)
                with patch('crbot.tactical_objective.phase_for',return_value='prepare'), \
                     patch.object(p,'candidates',return_value=roots), \
                     patch.object(p,'evaluate_root',side_effect=evaluate), \
                     patch.object(p,'refine_combinations',return_value=[combo]):
                    result=p.plan(self.scenario(3))
                self.assertEqual(result.status,expected)
                self.assertEqual(result.compute['tower_wait_guard']['changed'],expected=='ready')

    def test_tower_wait_guard_requires_real_affordable_mitigation(self):
        from crbot.tactical_objective import protect_towers_before_wait
        waiting=row(SimAction(),400,100)
        ineffective=row(SimAction('knight',0,.3,.72),390)
        effective=row(SimAction('musketeer',1,.3,.72),0)
        chosen,audit=protect_towers_before_wait(waiting,[waiting,ineffective,effective],self.scenario(3),self.kb)
        self.assertIs(chosen,waiting)
        self.assertFalse(audit['changed'])
        chosen,_=protect_towers_before_wait(waiting,[waiting,ineffective,effective],self.scenario(4),self.kb)
        self.assertIs(chosen,effective)

    def test_played_card_is_not_counted_as_still_in_hand(self):
        waiting = row(SimAction(),0)
        play = row(SimAction('musketeer',1,.3,.72),0)
        _, status = guard_development_reserve([waiting,play],self.scenario(),self.kb,'counterpush')
        self.assertNotIn('musketeer',status['options'][0]['retained_defenders'])
        self.assertIn('air',status['options'][0]['uncovered_layers'])

    def test_urgent_real_tower_mitigation_can_use_reserve(self):
        waiting = row(SimAction(),400)
        play = row(SimAction('knight',0,.3,.72),0)
        choices,status = guard_development_reserve([waiting,play],self.scenario(3),self.kb,'defend',urgent=True)
        self.assertIn(play,choices)
        self.assertTrue(status['options'][0]['tower_mitigation_override'])
        choices,_ = guard_development_reserve([row(SimAction(),0),play],self.scenario(3),self.kb,'defend',urgent=True)
        self.assertEqual(len(choices),1)

    def test_planner_waits_instead_of_spending_last_fees_on_high_scoring_development(self):
        p=PredictivePlanner(self.kb,dict(fast_defense=True,unified_tactics=True,development_reserve_gate=True))
        p.clock=lambda:0
        waiting=SimAction();play=SimAction('knight',0,.3,.72)
        with patch.object(p,'candidates',return_value=[waiting,play]), \
             patch.object(p,'evaluate_root',side_effect=lambda w,a,*args:row(a,0,100 if a.card_id else 0)), \
             patch.object(p,'refine_combinations',return_value=[]):
            result=p.plan(self.scenario())
        self.assertEqual(result.status,'wait')
        self.assertEqual(result.compute['development_reserve']['blocked'],1)

    def test_near_cap_still_develops_when_next_defense_is_affordable(self):
        p=PredictivePlanner(self.kb,dict(fast_defense=True,unified_tactics=True,development_reserve_gate=True))
        p.clock=lambda:0
        waiting=SimAction();play=SimAction('knight',0,.3,.72)
        with patch.object(p,'candidates',return_value=[waiting,play]), \
             patch.object(p,'evaluate_root',side_effect=lambda w,a,*args:row(a,0,-1 if a.card_id else 0)), \
             patch.object(p,'refine_combinations',return_value=[]):
            result=p.plan(self.scenario(10))
        self.assertEqual(result.status,'ready')
        self.assertEqual(result.action.card_id,'knight')

    def test_future_combo_cannot_spend_reserve_using_unexecuted_followup(self):
        waiting=row(SimAction(),400)
        play=row(SimAction('knight',0,.3,.72),400)
        combo=row(SimAction('knight',0,.3,.72),0,100)
        combo['planned_cost']=7
        choices,status=guard_development_reserve([waiting,play,combo],self.scenario(3),self.kb,'defend',
                                                urgent=True,immediate_candidates=[waiting,play])
        self.assertEqual(choices,[waiting])
        self.assertFalse(any(o['tower_mitigation_override'] for o in status['options']))

    def test_overflow_does_not_reintroduce_an_unfunded_next_defense(self):
        p=PredictivePlanner(self.kb,dict(fast_defense=True,unified_tactics=True,development_reserve_gate=True))
        p.clock=lambda:0
        roots=[SimAction(),SimAction('musketeer',0,.3,.72),SimAction('mega_knight',1,.3,.72)]
        with patch.object(p,'candidates',return_value=roots), \
             patch.object(p,'evaluate_root',side_effect=lambda w,a,*args:row(a,0,-1 if a.card_id else 0)), \
             patch.object(p,'refine_combinations',return_value=[]):
            result=p.plan(world(elixir=10,tracks=(),hand=((0,'musketeer'),(1,'mega_knight'))))
        self.assertEqual(result.status,'wait')
        self.assertEqual(result.compute['development_reserve']['blocked'],2)

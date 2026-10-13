"""Bounded temporal damage estimates, separate from timestamped visual HP."""
import time


class CombatHealth:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = None
        self.at = None
        self.revision = None
        self.identities = {}
        self.projected_state = None

    def update(self, sim, world, budget_s=.008):
        started = time.perf_counter()
        projected = {}
        complete = False
        self.projected_state = None
        if (self.state is not None and self.at is not None and
                world.revision > self.revision and 0 <= world.at-self.at <= 1.5):
            trial = self.state.clone()
            try:
                sim.advance(trial, world.at-self.at, deadline=started+budget_s, clock=time.perf_counter)
                projected = {e.track_id: e.hp/e.spec.hp for e in trial.entities if e.track_id is not None}
                complete = True
                self.projected_state = trial
            except TimeoutError:
                pass  # Never publish a partially advanced estimate.
        healing = any('heal' in t.card_id or t.card_id == 'battle_healer' for t in world.tracks)
        rows = []
        for t in world.tracks:
            fresh = t.hp_fraction is not None and (t.hp_observed_at is None or world.at-t.hp_observed_at <= .6)
            age = max(0., world.at-(t.hp_observed_at if t.hp_observed_at is not None else world.at))
            anchor = t.hp_fraction
            same = self.identities.get(t.track_id) == (t.card_id, t.side, t.first_seen)
            nominal = projected.get(t.track_id, 0.) if complete and same else None
            if fresh:
                low = high = nominal = anchor
                source = 'visual_health'
            else:
                # Widen stale bars instead of resetting injured bodies to full HP.
                high = min(1., anchor+.05*age) if anchor is not None and age <= 3 else 1.
                if healing:
                    high = 1.
                nominal = high if nominal is None else min(high, max(.01, nominal))
                if complete and same and not healing and not t.hypotheses:
                    high = min(high, max(.05, nominal+.15))
                # Seeing a body is not proof of its predicted death; uncertain
                # health must never silently remove a currently observed enemy.
                low = max(0., nominal-.15-.05*age)
                source = 'combat_projection' if complete and same else 'health_interval'
            t.hp_estimate = nominal
            t.hp_estimate_low, t.hp_estimate_high = low, high
            t.hp_estimate_source = source
            rows.append(dict(track_id=t.track_id,card_id=t.card_id,side=t.side,
                             observed_hp=anchor,observed_at=t.hp_observed_at,
                             estimate=nominal,low=low,high=high,source=source))
        return dict(complete=complete,elapsed_ms=round((time.perf_counter()-started)*1000,3),units=rows)

    def remember(self, state, world):
        self.state = state.clone()
        # Keep attack clocks and in-flight shots across observations. Rebuilding
        # towers with a ready gun every frame would fabricate extra damage.
        prior = self.projected_state
        def key(e):
            return ('track',e.track_id,e.spec.name,e.side) if e.track_id is not None else (('tower',e.side,e.x,e.y) if e.tower else None)
        if prior is not None:
            previous={key(e):e for e in prior.entities if key(e) is not None}
            current={key(e):e for e in self.state.entities if key(e) is not None}
            remap={old.uid:current[k].uid for k,old in previous.items() if k in current}
            for k,e in current.items():
                old=previous.get(k)
                if old is None:
                    continue
                e.ready_at=max(0.,old.ready_at-prior.time)
                e.stunned_until=max(0.,old.stunned_until-prior.time)
                e.target=remap.get(old.target)
                e.winding_target=remap.get(old.winding_target)
                e.shield=old.shield
            for impact in prior.impacts:
                at,side,target,x,y,damage,radius,air,mult,*control=impact
                if target is not None and target not in remap and radius<=0:
                    continue
                if len(control)>2:
                    control[2]=remap.get(control[2])
                self.state.impacts.append((max(0.,at-prior.time),side,remap.get(target),x,y,damage,radius,air,mult,*control))
        for e in self.state.entities:
            track = next((t for t in world.tracks if t.track_id == e.track_id), None)
            if track is not None and track.hp_estimate is not None:
                e.hp = e.spec.hp*track.hp_estimate
        self.at, self.revision = world.at, world.revision
        self.identities = {t.track_id: (t.card_id,t.side,t.first_seen) for t in world.tracks}

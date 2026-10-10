"""Position-aware, whole-battle-balanced outcome baseline for P1.

Only observed actions are labels. This estimates outcome association, not a
counterfactual advantage; unseen cards, positions and states abstain.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np

LEGACY_SCHEMA = "observed_state_action_knn_v1"
SCHEMA = "observed_state_action_knn_v2"
MIN_SUPPORT = 3
DEFAULT_RECIPE = dict(neighbors=8, state_radius=.3, position_radius=.15, shrinkage=32.)


def _recipe(value):
    recipe = {**DEFAULT_RECIPE, **value}
    if set(recipe) != set(DEFAULT_RECIPE):
        raise ValueError("invalid action value recipe keys")
    if (isinstance(recipe["neighbors"], bool) or not isinstance(recipe["neighbors"], (int, np.integer))
            or not MIN_SUPPORT <= recipe["neighbors"] <= 64):
        raise ValueError("invalid action value neighbors")
    for name in ("state_radius", "position_radius", "shrinkage"):
        number = float(recipe[name])
        if not math.isfinite(number) or number < 0 or (name != "shrinkage" and number == 0):
            raise ValueError("invalid action value recipe")
        recipe[name] = number
    return recipe


def _prior(y, groups):
    # Every battle has one vote, regardless of its number of recorded actions.
    means = [float(np.mean(y[groups == group])) for group in np.unique(groups)]
    return float(np.mean(means)) if means else .5


class ActionValueHead:
    def __init__(self, x, xy, y, cards, groups, *, enabled=False, recipe=None):
        self.x = np.asarray(x, dtype=np.float32)
        self.xy = np.asarray(xy, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.cards, self.groups = np.asarray(cards), np.asarray(groups)
        n = len(self.y)
        if (self.x.ndim != 2 or self.y.ndim != 1 or self.xy.shape != (n, 2)
                or len(self.x) != n or self.cards.shape != (n,) or self.groups.shape != (n,)
                or not all(np.isfinite(a).all() for a in (self.x, self.xy, self.y))
                or np.any((self.y < 0) | (self.y > 1))
                or np.any((self.xy < 0) | (self.xy > 1))):
            raise ValueError("invalid action value arrays")
        self.enabled = bool(enabled and n)
        self.recipe = _recipe(recipe) if recipe is not None else None
        self.prior = _prior(self.y, self.groups)
        self.indices = {str(c): np.flatnonzero(self.cards == c) for c in np.unique(self.cards)}

    def predict(self, card_id, feature, point):
        indices = self.indices.get(card_id)
        if indices is None or not len(indices):
            return None
        feature = np.asarray(feature, dtype=np.float32)
        point = np.asarray(point, dtype=np.float32)
        if (feature.shape != self.x.shape[1:] or point.shape != (2,)
                or not np.isfinite(feature).all() or not np.isfinite(point).all()):
            return None
        state_distance = np.sqrt(np.mean((self.x[indices] - feature) ** 2, axis=1))
        position_distance = np.linalg.norm(self.xy[indices] - point, axis=1)
        recipe = self.recipe
        state_radius = recipe["state_radius"] if recipe else .5
        position_radius = recipe["position_radius"] if recipe else .15
        supported = (state_distance <= state_radius) & (position_distance <= position_radius)
        indices = indices[supported]
        distances = ((state_distance / state_radius + position_distance / position_radius)
                     if recipe else (state_distance + 4 * position_distance))[supported]
        return self._estimate(indices, distances)

    def _estimate(self, indices, distances):
        # At most one vote per battle; repeated actions cannot create confidence.
        votes, weights, seen = [], [], set()
        for i in np.argsort(distances, kind="stable"):
            group = str(self.groups[indices[i]])
            if group in seen:
                continue
            seen.add(group)
            votes.append(float(self.y[indices[i]]))
            weights.append(float(np.exp(-distances[i])) if self.recipe else 1.)
            if len(votes) == (self.recipe["neighbors"] if self.recipe else 8):
                break
        if len(votes) < MIN_SUPPORT:
            return None
        mass = sum(weights)
        shrinkage = self.recipe["shrinkage"] if self.recipe else 0.
        value = (float(np.dot(votes, weights)) + shrinkage * self.prior) / (mass + shrinkage)
        return {"value": value, "support_battles": len(votes),
                "evidence_weight": mass, "prior": self.prior,
                "shrinkage_fraction": shrinkage / (mass + shrinkage)}

    def arrays(self):
        arrays = dict(action_value_schema=np.asarray([SCHEMA if self.recipe else LEGACY_SCHEMA]), action_value_x=self.x,
                    action_value_xy=self.xy, action_value_y=self.y,
                    action_value_cards=self.cards, action_value_groups=self.groups,
                    action_value_enabled=np.asarray([int(self.enabled)], dtype=np.int8))
        if self.recipe:
            for key, value in self.recipe.items():
                arrays["action_value_" + key] = np.asarray([value])
        return arrays

    @classmethod
    def load(cls, data):
        if "action_value_schema" not in data:
            return None
        schema = str(data["action_value_schema"][0])
        if schema not in (LEGACY_SCHEMA, SCHEMA):
            raise ValueError("unsupported action value schema")
        recipe = None
        if schema == SCHEMA:
            recipe = {}
            for key in DEFAULT_RECIPE:
                value = data["action_value_" + key]
                if np.asarray(value).shape != (1,):
                    raise ValueError("invalid action value recipe array")
                number = float(value[0])
                if key == "neighbors":
                    if not math.isfinite(number) or number != int(number):
                        raise ValueError("invalid action value neighbors")
                    number = int(number)
                recipe[key] = number
        return cls(*(data["action_value_" + k].copy() for k in ("x", "xy", "y", "cards", "groups")),
                   enabled=bool(data["action_value_enabled"][0]), recipe=recipe)


def fit(actions, feature, *, recipe=None):
    return ActionValueHead(
        np.asarray([feature(a) for a in actions], dtype=np.float32).reshape(len(actions), -1)
        if actions else np.empty((0, 0)),
        np.asarray([a.deploy_point for a in actions], dtype=np.float32).reshape(-1, 2),
        [a.target for a in actions], [a.card_id for a in actions], [a.group_id for a in actions], recipe=recipe)


def evaluate(train, validation, feature, *, recipe=None):
    head = fit(train, feature, recipe=recipe)
    train_groups = {a.group_id for a in train}
    if train_groups & {a.group_id for a in validation}:
        raise ValueError("action value evaluation battle leakage")
    prior = head.prior
    errors, baseline, supported = {}, {}, 0
    card_rows = {}
    support_histogram = {}
    unsupported = dict(unseen_card=0, insufficient_similar_battles=0)
    for a in validation:
        prediction = head.predict(a.card_id, feature(a), a.deploy_point)
        supported += prediction is not None
        value = prediction["value"] if prediction is not None else prior
        support = prediction["support_battles"] if prediction else 0
        support_histogram[str(support)] = support_histogram.get(str(support), 0) + 1
        if not prediction:
            reason = "unseen_card" if a.card_id not in head.indices else "insufficient_similar_battles"
            unsupported[reason] += 1
        card = card_rows.setdefault(a.card_id, dict(errors={}, baseline={}, actions=0, supported=0))
        card["actions"] += 1
        card["supported"] += prediction is not None
        card["errors"].setdefault(a.group_id, []).append((value - a.target) ** 2)
        card["baseline"].setdefault(a.group_id, []).append((prior - a.target) ** 2)
        errors.setdefault(a.group_id, []).append((value - a.target) ** 2)
        baseline.setdefault(a.group_id, []).append((prior - a.target) ** 2)
    brier = float(np.mean([np.mean(v) for v in errors.values()])) if errors else 1.
    base = float(np.mean([np.mean(v) for v in baseline.values()])) if baseline else 0.
    coverage = supported / max(1, len(validation))
    passed = (len(errors) >= 8 and len({a.target for a in validation}) == 2
              and coverage >= .5 and brier < base - 1e-6)
    reasons = []
    if len(errors) < 8:
        reasons.append("insufficient_validation_battles")
    if len({a.target for a in validation}) != 2:
        reasons.append("validation_requires_both_outcomes")
    if coverage < .5:
        reasons.append("insufficient_supported_actions")
    if brier >= base - 1e-6:
        reasons.append("not_better_than_battle_prior")
    cards = {}
    for name, rows in sorted(card_rows.items()):
        err = float(np.mean([np.mean(v) for v in rows["errors"].values()]))
        ref = float(np.mean([np.mean(v) for v in rows["baseline"].values()]))
        cards[name] = dict(actions=rows["actions"], battles=len(rows["errors"]),
                           coverage=rows["supported"] / rows["actions"], brier=err,
                           prior_brier=ref, brier_gain=ref-err)
    return dict(passed=passed, battles=len(errors), actions=len(validation), coverage=coverage,
                brier=brier, prior_brier=base, brier_gain=base-brier, prior=prior,
                rejection_reasons=reasons, by_card=cards, support_histogram=support_histogram,
                unsupported=unsupported, train_battles=sorted(train_groups),
                validation_battles=sorted(errors))


def _inner_split(actions, seed):
    outcomes = {a.group_id: a.target for a in actions}
    held_out = set()
    for outcome in sorted(set(outcomes.values())):
        groups = sorted((g for g, y in outcomes.items() if y == outcome),
                        key=lambda g: hashlib.sha256(f"{seed}:action-value:{g}".encode()).digest())
        if len(groups) < 2:
            return actions, []
        held_out.update(groups[:max(1, math.ceil(len(groups) * .25))])
    return ([a for a in actions if a.group_id not in held_out],
            [a for a in actions if a.group_id in held_out])


def select_recipe(actions, feature, *, seed):
    """Tune only inside outer training battles; no final validation labels."""
    train, validation = _inner_split(actions, seed)
    report = dict(train_battles=sorted({a.group_id for a in train}),
                  validation_battles=sorted({a.group_id for a in validation}), trials=[])
    if len(report["validation_battles"]) < 8:
        report.update(reason="insufficient_inner_battles", selected=dict(DEFAULT_RECIPE))
        return dict(DEFAULT_RECIPE), report
    # Materialize features once. Each radius/neighbors combination shares predictions
    # across shrinkage choices, so parameter search does not repeat distance scans.
    head = fit(train, feature, recipe=DEFAULT_RECIPE)
    queries = []
    for a in validation:
        indices = head.indices.get(a.card_id, np.asarray([], dtype=int))
        state = np.sqrt(np.mean((head.x[indices] - feature(a)) ** 2, axis=1))
        position = np.linalg.norm(head.xy[indices] - a.deploy_point, axis=1)
        queries.append((a, indices, state, position))
    scored = []
    for neighbors, state_radius, position_radius in itertools.product((4, 8, 16), (.15, .3, .5), (.08, .15)):
        predictions = []
        head.recipe = dict(neighbors=neighbors, state_radius=state_radius,
                           position_radius=position_radius, shrinkage=0.)
        for a, indices, state, position in queries:
            mask = (state <= state_radius) & (position <= position_radius)
            prediction = head._estimate(indices[mask], (state/state_radius + position/position_radius)[mask])
            predictions.append((a, prediction))
        for shrinkage in (0., 8., 32.):
            errors, baseline, supported = {}, {}, 0
            for a, prediction in predictions:
                value = head.prior
                if prediction:
                    mass = prediction["evidence_weight"]
                    value = (prediction["value"] * mass + shrinkage * head.prior) / (mass + shrinkage)
                    supported += 1
                errors.setdefault(a.group_id, []).append((value-a.target)**2)
                baseline.setdefault(a.group_id, []).append((head.prior-a.target)**2)
            brier = float(np.mean([np.mean(v) for v in errors.values()]))
            base = float(np.mean([np.mean(v) for v in baseline.values()]))
            recipe = dict(neighbors=neighbors, state_radius=state_radius,
                          position_radius=position_radius, shrinkage=shrinkage)
            coverage = supported / len(validation)
            report["trials"].append(dict(recipe=recipe, brier=brier, prior_brier=base, coverage=coverage))
            if coverage >= .5:
                # Deterministic ties prefer stronger shrinkage and narrower support.
                scored.append(((brier, -shrinkage, state_radius, position_radius, neighbors), recipe))
    selected = min(scored, key=lambda row: row[0])[1] if scored else dict(DEFAULT_RECIPE)
    report.update(reason="inner_validation" if scored else "insufficient_inner_support", selected=selected)
    return selected, report


def train_head(actions, train, validation, temporal_train, temporal_validation, feature,
               *, require_temporal=False, seed=20260903):
    validation_groups = {a.group_id for a in validation}
    temporal_groups = {a.group_id for a in temporal_validation} if require_temporal else set()
    if {a.group_id for a in train} & validation_groups:
        raise ValueError("action value evaluation battle leakage")
    if require_temporal and {a.group_id for a in temporal_train} & temporal_groups:
        raise ValueError("action value evaluation battle leakage")
    common = ({a.group_id for a in temporal_train} if require_temporal else {a.group_id for a in train})
    tuning = [a for a in train if a.group_id in common and a.group_id not in validation_groups | temporal_groups]
    recipe, selection = select_recipe(tuning, feature, seed=seed)
    random = evaluate(train, validation, feature, recipe=recipe)
    temporal = evaluate(temporal_train, temporal_validation, feature, recipe=recipe) if require_temporal else None
    legacy = dict(random=evaluate(train, validation, feature),
                  temporal=evaluate(temporal_train, temporal_validation, feature) if require_temporal else None)
    head = fit(actions, feature, recipe=recipe)
    head.enabled = random["passed"] and (not require_temporal or temporal["passed"])
    return head, dict(schema=SCHEMA, enabled=head.enabled, random=random, temporal=temporal,
                      recipe=recipe, selection=selection, legacy=legacy,
                      label="observed_battle_outcome", counterfactual_ground_truth=False,
                      wait_supported=False, minimum_support_battles=MIN_SUPPORT,
                      fit_battles=sorted({a.group_id for a in actions}))

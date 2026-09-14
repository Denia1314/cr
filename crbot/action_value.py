"""Position-aware, whole-battle-balanced outcome baseline for P1.

Only observed actions are labels. This estimates outcome association, not a
counterfactual advantage; unseen cards, positions and states abstain.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "observed_state_action_knn_v1"
MIN_SUPPORT = 3


class ActionValueHead:
    def __init__(self, x, xy, y, cards, groups, *, enabled=False):
        self.x = np.asarray(x, dtype=np.float32)
        self.xy = np.asarray(xy, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.cards, self.groups = np.asarray(cards), np.asarray(groups)
        n = len(self.y)
        if (self.x.ndim != 2 or self.xy.shape != (n, 2)
                or len(self.x) != n or self.cards.shape != (n,) or self.groups.shape != (n,)
                or not all(np.isfinite(a).all() for a in (self.x, self.xy, self.y))
                or np.any((self.y < 0) | (self.y > 1))
                or np.any((self.xy < 0) | (self.xy > 1))):
            raise ValueError("invalid action value arrays")
        self.enabled = bool(enabled and n)
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
        supported = (state_distance <= .5) & (position_distance <= .15)
        indices = indices[supported]
        distances = (state_distance + 4 * position_distance)[supported]
        # At most one vote per battle; repeated actions cannot create confidence.
        votes, seen = [], set()
        for i in np.argsort(distances, kind="stable"):
            group = str(self.groups[indices[i]])
            if group in seen:
                continue
            seen.add(group)
            votes.append(float(self.y[indices[i]]))
            if len(votes) == 8:
                break
        if len(votes) < MIN_SUPPORT:
            return None
        return {"value": float(np.mean(votes)), "support_battles": len(votes)}

    def arrays(self):
        return dict(action_value_schema=np.asarray([SCHEMA]), action_value_x=self.x,
                    action_value_xy=self.xy, action_value_y=self.y,
                    action_value_cards=self.cards, action_value_groups=self.groups,
                    action_value_enabled=np.asarray([int(self.enabled)], dtype=np.int8))

    @classmethod
    def load(cls, data):
        if "action_value_schema" not in data:
            return None
        if str(data["action_value_schema"][0]) != SCHEMA:
            raise ValueError("unsupported action value schema")
        return cls(*(data["action_value_" + k].copy() for k in ("x", "xy", "y", "cards", "groups")),
                   enabled=bool(data["action_value_enabled"][0]))


def fit(actions, feature):
    return ActionValueHead(
        np.asarray([feature(a) for a in actions], dtype=np.float32).reshape(len(actions), -1)
        if actions else np.empty((0, 0)),
        np.asarray([a.deploy_point for a in actions], dtype=np.float32).reshape(-1, 2),
        [a.target for a in actions], [a.card_id for a in actions], [a.group_id for a in actions])


def evaluate(train, validation, feature):
    head = fit(train, feature)
    train_groups = {a.group_id for a in train}
    if train_groups & {a.group_id for a in validation}:
        raise ValueError("action value evaluation battle leakage")
    outcomes = {}
    for a in train:
        outcomes.setdefault(a.group_id, []).append(a.target)
    prior = float(np.mean([np.mean(v) for v in outcomes.values()])) if train else .5
    errors, baseline, supported = {}, {}, 0
    for a in validation:
        prediction = head.predict(a.card_id, feature(a), a.deploy_point)
        supported += prediction is not None
        value = prediction["value"] if prediction is not None else prior
        errors.setdefault(a.group_id, []).append((value - a.target) ** 2)
        baseline.setdefault(a.group_id, []).append((prior - a.target) ** 2)
    brier = float(np.mean([np.mean(v) for v in errors.values()])) if errors else 1.
    base = float(np.mean([np.mean(v) for v in baseline.values()])) if baseline else 0.
    coverage = supported / max(1, len(validation))
    passed = (len(errors) >= 8 and len({a.target for a in validation}) == 2
              and coverage >= .5 and brier < base - 1e-6)
    return dict(passed=passed, battles=len(errors), actions=len(validation), coverage=coverage,
                brier=brier, prior_brier=base, train_battles=sorted(train_groups),
                validation_battles=sorted(errors))


def train_head(actions, train, validation, temporal_train, temporal_validation, feature,
               *, require_temporal=False):
    random = evaluate(train, validation, feature)
    temporal = evaluate(temporal_train, temporal_validation, feature) if require_temporal else None
    head = fit(actions, feature)
    head.enabled = random["passed"] and (not require_temporal or temporal["passed"])
    return head, dict(schema=SCHEMA, enabled=head.enabled, random=random, temporal=temporal,
                      label="observed_battle_outcome", counterfactual_ground_truth=False,
                      wait_supported=False, minimum_support_battles=MIN_SUPPORT,
                      fit_battles=sorted({a.group_id for a in actions}))

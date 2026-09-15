"""Finite deployment domains, with no top-k or spatial suppression."""
from __future__ import annotations
import math


def deployment_domain(sim, state, card_id, side, *, step=.5, image_size=None):
    left,top,right,bottom=sim.geometry.bounds
    if image_size is not None:
        width, height = image_size
        if width < 2 or height < 2:
            raise ValueError("invalid image size")
        xs = [x/width for x in range(math.ceil(left*width),math.floor(right*width)+1)]
        ys = [y/height for y in range(math.ceil(top*height),math.floor(bottom*height)+1)]
    else:
        if not math.isfinite(step) or not .125 <= step <= 1:
            raise ValueError("placement grid step must be between .125 and 1 tile")
        xs = [sim.screen(min(18,i*step),0)[0] for i in range(math.ceil(18/step)+1)]
        ys = [sim.screen(0,min(32,i*step))[1] for i in range(math.ceil(32/step)+1)]
    points = [(x,y) for x in xs for y in ys]
    batch = getattr(sim, "placement_batch", None)
    if batch is not None and hasattr(batch, "legal_points"):
        return batch.legal_points(sim,state,card_id,side,points)
    return [p for p in points if sim.legal_placement(state,card_id,side,*p)]

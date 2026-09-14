"""Finite deployment domains, with no top-k or spatial suppression."""
from __future__ import annotations
import math


def deployment_domain(sim, state, card_id, side, *, step=.5, image_size=None):
    if image_size is not None:
        width, height = image_size
        if width < 2 or height < 2:
            raise ValueError("invalid image size")
        xs = [x/width for x in range(math.ceil(.05*width),math.floor(.95*width)+1)]
        ys = [y/height for y in range(math.ceil(.18*height),math.floor(.82*height)+1)]
    else:
        if not math.isfinite(step) or not .125 <= step <= 1:
            raise ValueError("placement grid step must be between .125 and 1 tile")
        xs = [round(.05 + min(18,i*step)*.05,12) for i in range(math.ceil(18/step)+1)]
        ys = [round(.18 + min(32,i*step)*.02,12) for i in range(math.ceil(32/step)+1)]
    return [(x,y) for x in xs for y in ys if sim.legal_placement(state,card_id,side,x,y)]

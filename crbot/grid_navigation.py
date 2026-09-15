"""Bounded, cached 18 x 32 navigation shared by simulation and inspection."""
import heapq
import math
from functools import lru_cache


def cell(x, y):
    return max(0, min(17, int(x))), max(0, min(31, int(y)))


def passable(x, y, bridges, blocked, air=False):
    if not (0 <= x < 18 and 0 <= y < 32):
        return False
    return air or ((x, y) not in blocked and
                   (y not in (15, 16) or any(abs(x+.5-b) <= 1 for b in bridges)))


@lru_cache(maxsize=4096)
def _route(start, goal, bridges, blocked, air):
    blocked = frozenset(blocked) - {start, goal}
    queue = [(0, start)]; costs = {start: 0}; parents = {}
    while queue:
        _, here = heapq.heappop(queue)
        if here == goal:
            path = [here]
            while here in parents:
                here = parents[here]; path.append(here)
            return tuple(reversed(path))
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            nxt = here[0]+dx, here[1]+dy
            if not passable(*nxt, bridges, blocked, air):
                continue
            if dx and dy and (not passable(here[0]+dx, here[1], bridges, blocked, air)
                              or not passable(here[0], here[1]+dy, bridges, blocked, air)):
                continue
            cost = costs[here] + math.hypot(dx,dy)
            if cost >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = cost; parents[nxt] = here
            heapq.heappush(queue,(cost+math.hypot(nxt[0]-goal[0],nxt[1]-goal[1]),nxt))
    return ()


def obstacles(state, mover, target):
    blocked = set()
    if mover.spec.air:
        return ()
    for entity in state.entities:
        if entity.uid in (mover.uid,target.uid) or entity.hp <= 0 or not (entity.tower or entity.spec.building):
            continue
        radius = entity.spec.radius + min(.4,mover.spec.radius)
        for x in range(max(0,int(entity.x-radius)),min(18,int(entity.x+radius)+1)):
            for y in range(max(0,int(entity.y-radius)),min(32,int(entity.y+radius)+1)):
                if math.hypot(x+.5-entity.x,y+.5-entity.y) < radius:
                    blocked.add((x,y))
    return tuple(sorted(blocked))


def route(state, mover, target, geometry):
    if mover.spec.air:
        return [(mover.x,mover.y),(target.x,target.y)]
    blocked=obstacles(state,mover,target)
    if (mover.y-16)*(target.y-16)>=0:
        steps=max(1,math.ceil(math.hypot(target.x-mover.x,target.y-mover.y)*3))
        if all(passable(*cell(mover.x+(target.x-mover.x)*i/steps,
                             mover.y+(target.y-mover.y)*i/steps),geometry.bridges,blocked)
               for i in range(1,steps+1)):
            return [(mover.x,mover.y),(target.x,target.y)]
    nodes = _route(cell(mover.x,mover.y),cell(target.x,target.y),tuple(geometry.bridges),
                   blocked,False)
    if not nodes:
        return []
    return [(mover.x,mover.y)]+[(x+.5,y+.5) for x,y in nodes[1:-1]]+[(target.x,target.y)]


def move(state, mover, target, geometry, distance):
    points = route(state,mover,target,geometry)
    if len(points)<2:
        return 0.
    tx,ty=points[1]; length=math.hypot(tx-mover.x,ty-mover.y)
    if not length:
        return 0.
    step=min(distance,length)
    nx,ny=mover.x+(tx-mover.x)/length*step,mover.y+(ty-mover.y)/length*step
    # Soft local avoidance; allow escape from an initial detector overlap. This
    # is not the game's exact mass/push system and never changes observed boxes.
    for other in state.entities:
        if other.uid in (mover.uid,target.uid) or other.hp<=0 or other.spec.air!=mover.spec.air:
            continue
        radius=min(.55,mover.spec.radius+other.spec.radius)
        old=math.hypot(mover.x-other.x,mover.y-other.y)
        new=math.hypot(nx-other.x,ny-other.y)
        if new < radius and new < old:
            step *= .5
            nx,ny=mover.x+(tx-mover.x)/length*step,mover.y+(ty-mover.y)/length*step
    mover.x,mover.y=nx,ny
    return step

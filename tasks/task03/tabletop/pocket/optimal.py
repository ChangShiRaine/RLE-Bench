"""Exact QTM distance by bidirectional BFS, modulo whole-cube rotation."""
from functools import lru_cache
import numpy as np
from . import metrics
from . import metrics as cube

R=cube.ROTATIONS.astype(int);lookup={tuple(x.flat):i for i,x in enumerate(R)}
identity=lookup[tuple(np.eye(3,dtype=int).flat)]
mult=[[lookup[tuple((a@b).flat)] for b in R] for a in R]
inverse=[lookup[tuple(a.T.flat)] for a in R]
positions=np.einsum('rij,bj->bri',R,metrics.POSITIONS)
moves=[(a,s,t) for a in range(3) for s in (-1,1) for t in (-1,1)]
turn=[]
for a,s,t in moves:
 d=np.eye(3,dtype=int)[a]*s
 cross=np.array([[0,-d[2],d[1]],[d[2],0,-d[0]],[-d[1],d[0],0]])
 mat=np.outer(d,d)+cross
 if t==-1:mat=mat.T
 turn.append(lookup[tuple(mat.flat)])
selected=[[[bool(positions[b,r,a]*s>.5) for r in range(24)] for b in range(8)] for a,s,t in moves]
def encode(rot):return tuple(lookup[tuple(x.astype(int).flat)] for x in rot)
def advance(state,m):
 moved=tuple(mult[turn[m]][r] if selected[m][b][r] else r for b,r in enumerate(state))
 inv=inverse[moved[0]]
 return tuple(mult[inv][r] for r in moved)

@lru_cache(maxsize=128)
def _distance(start):
    goal = (identity,) * 8
    if start == goal:
        return 0
    left, right = {start}, {goal}
    seen_left, seen_right = set(left), set(right)
    depth = 0
    while left and right:
        if len(left) > len(right):
            left, right = right, left
            seen_left, seen_right = seen_right, seen_left
        next_front = set()
        depth += 1
        for state in sorted(left):
            for move in range(len(moves)):
                nxt = advance(state, move)
                if nxt in seen_right:
                    return depth
                if nxt not in seen_left:
                    seen_left.add(nxt)
                    next_front.add(nxt)
        left = next_front
    raise ValueError("Unreachable cube state")


def optimal_qtm(rotations):
    state = encode(rotations)
    inv = inverse[state[0]]
    return _distance(tuple(mult[inv][r] for r in state))

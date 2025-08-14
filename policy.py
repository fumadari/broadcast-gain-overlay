
from __future__ import annotations
import numpy as np

class MoveWaitPolicy:
    """
    Simple stochastic policy producing logits for actions {move, wait}.
    Base logit for 'move' increases when far from the intersection to create demand spikes.
    Gain g (>=0) multiplies ONLY the move logit (move/wait gating).
    """
    def __init__(self, base_scale=1.0):
        self.base_scale = base_scale

    def base_logits(self, obs_row):
        # obs_row = (dist_to_intersection, at_intersection_flag)
        dist = obs_row[0]; at_inter = obs_row[1] > 0.5
        # encourage movement, but a bit less at the intersection
        base_move = self.base_scale * (1.0 + 0.1*dist) * (0.7 if at_inter else 1.0)
        base_wait = 0.0
        return np.array([base_move, base_wait], dtype=float)

    def action_probs(self, base_logits, g=1.0):
        # Apply gain to 'move' logit only
        move_logit = g * base_logits[0]
        wait_logit = base_logits[1]
        mx = max(move_logit, wait_logit)
        exps = np.exp([move_logit - mx, wait_logit - mx])
        p = exps / exps.sum()
        return p  # [p_move, p_wait]

    def sample(self, rng, base_logits, g=1.0):
        p = self.action_probs(base_logits, g=g)
        a = rng.choice(2, p=p)
        return ("move" if a==0 else "wait"), -np.log(p[a] + 1e-9)  # action, surprisal

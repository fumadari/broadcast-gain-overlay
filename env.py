
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ---- Simple plus-corridor grid with a 4-way intersection ----

DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}

OPP = {"N":"S","S":"N","E":"W","W":"E"}

@dataclass
class Robot:
    id: int
    r: int
    c: int
    heading: str   # one of N,S,E,W

class PlusCorridorEnv:
    """
    Grid with a horizontal and vertical corridor (width 1) crossing at center.
    Robots move along the corridors and bounce at ends (reverse heading).
    Actions: 'move' or 'wait'. Move attempts to step 1 cell along heading.
    Arbitration: if multiple robots target same cell (or try to occupy a cell with a waiter),
    exactly one wins (uniform random w.r.t. ids); others wait.
    """
    def __init__(self, H=24, W=24, n_robots=20, seed=0):
        self.H = H; self.W = W
        self.r_mid = H//2; self.c_mid = W//2
        self.rng = np.random.RandomState(seed)
        self.robots : List[Robot] = []
        self.t = 0
        self.n_robots = n_robots
        self._spawn()
        # occupancy mask for plotting (time x corridor_position)
        self.time = 0
        self.row_occ = []  # occupancy of central row (length W)
        self.col_occ = []  # occupancy of central col (length H)

    def _spawn(self):
        self.robots = []
        rid = 0
        # place robots on 4 arms, heading toward center initially
        arms = ["W->E","E->W","N->S","S->N"]
        per_arm = self.n_robots // 4
        spacing = max(1, (max(self.H,self.W)//2 - 2) // max(1, per_arm))
        # West arm (row r_mid, cols [1..c_mid-1]), heading East
        c = 1
        for k in range(per_arm):
            self.robots.append(Robot(rid, self.r_mid, c, "E")); rid += 1
            c += spacing
            if c >= self.c_mid: break
        # East arm
        c = self.W-2
        for k in range(per_arm):
            self.robots.append(Robot(rid, self.r_mid, c, "W")); rid += 1
            c -= spacing
            if c <= self.c_mid: break
        # North arm
        r = 1
        for k in range(per_arm):
            self.robots.append(Robot(rid, r, self.c_mid, "S")); rid += 1
            r += spacing
            if r >= self.r_mid: break
        # South arm
        r = self.H-2
        for k in range(per_arm):
            self.robots.append(Robot(rid, r, self.c_mid, "N")); rid += 1
            r -= spacing
            if r <= self.r_mid: break
        # If fewer than requested, fill from west arm
        while len(self.robots) < self.n_robots:
            c = self.rng.randint(1, self.c_mid-1)
            self.robots.append(Robot(rid, self.r_mid, c, "E")); rid += 1

    def _within_corridor(self, r, c):
        return (r == self.r_mid and 0 <= c < self.W) or (c == self.c_mid and 0 <= r < self.H)

    def _next_cell(self, robot: Robot):
        dr, dc = DIRS[robot.heading]
        return (robot.r + dr, robot.c + dc)

    def _bounce_if_edge(self, robot: Robot):
        # Reverse heading at corridor ends
        if robot.heading == "E" and robot.c == self.W-2:
            robot.heading = "W"
        elif robot.heading == "W" and robot.c == 1:
            robot.heading = "E"
        elif robot.heading == "S" and robot.r == self.H-2:
            robot.heading = "N"
        elif robot.heading == "N" and robot.r == 1:
            robot.heading = "S"

    def observe(self):
        # Minimal observation: distance to intersection, is at intersection, etc.
        obs = []
        for rb in self.robots:
            if rb.r == self.r_mid:
                dist = abs(rb.c - self.c_mid)
            else:
                dist = abs(rb.r - self.r_mid)
            at_inter = (rb.r == self.r_mid and rb.c == self.c_mid)
            obs.append((dist, int(at_inter)))
        return np.array(obs, dtype=float)  # shape (N,2)

    def step(self, intents: List[str]):
        """
        intents: list of 'move' or 'wait' proposed BEFORE arbitration.
        Returns:
            realized_moves: np.array of 0/1 indicating who actually moved.
            shared_reward: float in [0,1] == mean(realized_moves)
            info: dict with additional diagnostics + occupancy arrays
        """
        N = len(self.robots); assert N == len(intents)
        # record occupancy for plotting
        row = np.zeros(self.W, dtype=int)
        col = np.zeros(self.H, dtype=int)
        for rb in self.robots:
            if rb.r == self.r_mid: row[rb.c] = 1
            if rb.c == self.c_mid: col[rb.r] = 1
        self.row_occ.append(row)
        self.col_occ.append(col)

        # Map each target cell -> claimants
        claimants : Dict[Tuple[int,int], list] = {}
        targets = []
        for i, rb in enumerate(self.robots):
            self._bounce_if_edge(rb)
            if intents[i] == "move":
                nr, nc = self._next_cell(rb)
                if not self._within_corridor(nr, nc):
                    # treat off-corridor as blocked; convert to wait
                    nr, nc = rb.r, rb.c
            else:
                nr, nc = rb.r, rb.c
            targets.append((nr, nc))
            claimants.setdefault((nr,nc), []).append(i)

        realized = np.zeros(N, dtype=int)
        # Randomly choose a winner per contested cell
        for cell, idxs in claimants.items():
            if len(idxs) == 1:
                realized[idxs[0]] = 1 if intents[idxs[0]] == "move" else 0
            else:
                # choose one at random to succeed
                winner = int(self.rng.choice(idxs))
                if intents[winner] == "move":
                    realized[winner] = 1
                # others become waits (realized=0)

        # Apply movement for winners
        for i, rb in enumerate(self.robots):
            if realized[i] == 1:
                rb.r, rb.c = targets[i]  # step forward
        shared_reward = realized.mean()
        self.t += 1
        info = {
            "desired_move": np.array([1 if a=="move" else 0 for a in intents], dtype=int),
            "realized_move": realized.copy()
        }
        return realized, shared_reward, info

    def occupancy_mats(self):
        return np.array(self.row_occ, dtype=int), np.array(self.col_occ, dtype=int)

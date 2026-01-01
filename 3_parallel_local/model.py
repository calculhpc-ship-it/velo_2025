from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class State:
    """Represents the state of bikes at two stations."""
    mailly: int
    moulin: int
    unmet_mailly: int = 0
    unmet_moulin: int = 0

def step(state: State, p1: float, p2: float, rng: np.random.Generator, metrics: Dict[str, int] = None) -> State:
    """Simulate one time step of the bike-sharing system."""
    # Mailly -> Moulin
    if rng.random() < p1:
        if state.mailly > 0:
            state.mailly -= 1
            state.moulin += 1
        else:
            state.unmet_mailly += 1

    # Moulin -> Mailly
    if rng.random() < p2:
        if state.moulin > 0:
            state.moulin -= 1
            state.mailly += 1
        else:
            state.unmet_moulin += 1

    return state

def run_simulation(initial_mailly: int, initial_moulin: int, steps: int, p1: float, p2: float, seed: int):
    """Run a complete bike-sharing simulation with extended metrics."""
    rng = np.random.default_rng(seed)
    state = State(initial_mailly, initial_moulin)

    metrics = {
        'mailly': [],
        'moulin': [],
        'unmet_mailly': [],
        'unmet_moulin': [],
    }

    for _ in range(steps):
        state = step(state, p1, p2, rng)
        metrics['mailly'].append(state.mailly)
        metrics['moulin'].append(state.moulin)
        metrics['unmet_mailly'].append(state.unmet_mailly)
        metrics['unmet_moulin'].append(state.unmet_moulin)

    metrics['final_imbalance'] = state.mailly - state.moulin
    return metrics

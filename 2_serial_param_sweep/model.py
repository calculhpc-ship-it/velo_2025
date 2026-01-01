from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import pandas as pd

@dataclass
class State:
    """Represents the state of bikes at two stations."""
    mailly: int
    moulin: int
    unmet_mailly: int = 0
    unmet_moulin: int = 0

def step(
    state: State,
    p1: float,
    p2: float,
    rng: np.random.Generator,
    metrics: Dict[str, int],
) -> State:
    """Simulate one time step of the bike-sharing system."""
    # prob déplacement de Mailly-->Moulin
    if rng.random() < p1:
        if state.mailly > 0:
            state.mailly -= 1
            state.moulin += 1
        else:
            state.unmet_mailly += 1
    # prob déplacement velo Moulin-->Mailly
    if rng.random() < p2:
        if state.moulin > 0:
            state.moulin -= 1
            state.mailly += 1
        else:
            state.unmet_moulin += 1
    # Mettre à jour les metrics
    metrics['unmet_mailly'] = state.unmet_mailly
    metrics['unmet_moulin'] = state.unmet_moulin

    return state

def run_simulation(
    initial_mailly: int,
    initial_moulin: int,
    steps: int,
    p1: float,
    p2: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Run a complete bike-sharing simulation."""
    initial_mailly = int(initial_mailly)
    initial_moulin = int(initial_moulin)
    steps = int(steps)
    p1 = float(p1)
    p2 = float(p2)
    seed = int(seed)

    rng = np.random.default_rng(seed)

    # Initialisation 
    state = State(mailly=initial_mailly, moulin=initial_moulin)
    metrics = {
        'mailly': initial_mailly,
        'moulin': initial_moulin,
        'unmet_mailly': 0,
        'unmet_moulin': 0}
    
    time_series_data = {
        'time': [0],
        'mailly': [initial_mailly],
        'moulin': [initial_moulin]}
    # Boucle de simulation
    for t in range(1, steps + 1):
        metrics['mailly'] = state.mailly
        metrics['moulin'] = state.moulin

        state = step(state, p1, p2, rng, metrics)
        # sauvegarder des données 
        time_series_data['time'].append(t)
        time_series_data['mailly'].append(state.mailly)
        time_series_data['moulin'].append(state.moulin)

    metrics['final_imbalance'] = state.mailly - state.moulin
    df = pd.DataFrame(time_series_data)
    metrics['unmet_mailly'] = state.unmet_mailly
    metrics['unmet_moulin'] = state.unmet_moulin

    return df, metrics

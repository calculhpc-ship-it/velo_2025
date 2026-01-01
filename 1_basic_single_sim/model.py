from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import pandas as pd

@dataclass
class State:
    """Represent the state of bikes at two stations."""
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
    #prob de déplacement vélo de Mailly-->Moulin
    if rng.random() < p1:
        if state.mailly > 0:
            state.mailly -= 1
            state.moulin += 1
        else:
            state.unmet_mailly += 1
            metrics["unmet_mailly"] += 1
    #prob déplacement vélo de Moulin-->Mailly
    if rng.random() < p2:
        if state.moulin > 0:
            state.moulin -= 1
            state.mailly += 1
        else:
            state.unmet_moulin += 1
            metrics["unmet_moulin"] += 1

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
    # Initialisation
    rng = np.random.default_rng(seed)
    state = State(
        mailly=initial_mailly,
        moulin=initial_moulin,)
    metrics: Dict[str, int] = {
        "unmet_mailly": 0,
        "unmet_moulin": 0,}
    # sauvegarde des états 
    records = []

    for t in range(steps):
        records.append({
            "time": t,
            "mailly": state.mailly,
            "moulin": state.moulin,})

        state = step(
            state=state,
            p1=p1,
            p2=p2,
            rng=rng,
            metrics=metrics,)
    # Métriques finales
    metrics["mailly"] = state.mailly
    metrics["moulin"] = state.moulin
    metrics["final_imbalance"] = state.mailly - state.moulin
    df = pd.DataFrame.from_records(records)

    return df, metrics

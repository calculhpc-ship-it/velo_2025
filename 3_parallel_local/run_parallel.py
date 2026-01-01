import argparse
import time
from pathlib import Path
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from model import run_simulation

def parse_args():
    """Parse command line arguments for parallel parameter sweep."""
    parser = argparse.ArgumentParser(
    description="Bike simulation: sequential vs multiprocessing")
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()

def run_one_simulation(row):
    """Exécute une simulation"""
    metrics = run_simulation(
        initial_mailly=int(row["init_mailly"]),
        initial_moulin=int(row["init_moulin"]),
        steps=int(row["steps"]),
        p1=float(row["p1"]),
        p2=float(row["p2"]),
        seed=int(row["seed"])
    )
    # Fusionner paramètres et métriques finales
    result = {
        **row.to_dict(),
        "final_mailly": metrics["mailly"][-1],
        "final_moulin": metrics["moulin"][-1],
        "total_unmet_mailly": metrics["unmet_mailly"][-1],
        "total_unmet_moulin": metrics["unmet_moulin"][-1],
        "final_imbalance": metrics["final_imbalance"]
    }
    return result

def run_sequential(df):
    """Exécution séquentielle"""
    results = []
    start = time.perf_counter()
    for _, row in df.iterrows():
        results.append(run_one_simulation(row))
    duration = time.perf_counter() - start
    return results, duration

def process_worker(row, queue):
    """Worker pour exécuter la simulation dans un processus."""
    result = run_one_simulation(row)
    queue.put(result)  # communication inter-processus

def run_multiprocessing(df, workers):
    """Exécution multi-processus des simulations."""
    if workers == "auto":
        num_workers = mp.cpu_count()
    else:
        num_workers = int(workers)

    num_workers = min(num_workers, len(df))
    manager = mp.Manager()
    results_queue = manager.Queue()
    processes = []

    start = time.perf_counter()

    # Création d'un processus par jeu de paramètres
    for i in range(len(df)):
        p = mp.Process(
            target=process_worker,
            args=(df.iloc[i], results_queue)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    duration = time.perf_counter() - start
    return results, duration

def plot_performance(seq_time, mp_time, out_dir):
    """Graphique comparant séq/ multip avec speedup."""
    speedup = seq_time / mp_time
    labels = ["Sequential", "Multiprocessing"]
    times = [seq_time, mp_time]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, times)
    plt.ylabel("time (s)")
    plt.title("Execution time comparison")
    plt.text(
        0.5,
        max(times) * 0.9,
        f"Speedup = {speedup:.2f}x",
        ha="center",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7)
    )
    plt.tight_layout()
    plt.savefig(out_dir / "performance_comparison.png")
    plt.close()

def main():
    """Main function to run parallel parameter sweep using multiprocessing."""
    args = parse_args()
    df = pd.read_csv(args.params)
    args.out_dir.mkdir(exist_ok=True)

    seq_results, seq_time = run_sequential(df)
    print(f"Sequential time: {seq_time:.3f}s")

    mp_results, mp_time = run_multiprocessing(df, args.workers)
    print(f"Multiprocessing time: {mp_time:.3f}s")

    speedup = seq_time / mp_time
    print(f"Speedup: {speedup:.2f}x")
    print(
        "Note: le multiprocessing évite le GIL mais introduit un surcoût "
        "lié à la création des processus et à la communication via Queue.")

    pd.DataFrame(seq_results).to_csv(
        args.out_dir / "metrics_sequential.csv", index=False
    )
    pd.DataFrame(mp_results).to_csv(
        args.out_dir / "metrics_multiprocessing.csv", index=False )
    plot_performance(seq_time, mp_time, args.out_dir)

if __name__ == "__main__":
    main()

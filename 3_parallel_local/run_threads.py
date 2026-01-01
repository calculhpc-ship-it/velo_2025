import argparse
import time
import threading
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from model import run_simulation  

# Données partagées protégées par lock
global_stats = []
stats_lock = threading.Lock()

def parse_args():
    """Parse command line arguments for parallel parameter sweep."""
    parser = argparse.ArgumentParser(description="Bike simulation: sequential vs threading")
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
    # fusionner paramètres et metrics finales
    result = {
        **row.to_dict(),
        'final_mailly': metrics['mailly'][-1],
        'final_moulin': metrics['moulin'][-1],
        'total_unmet_mailly': metrics['unmet_mailly'][-1],
        'total_unmet_moulin': metrics['unmet_moulin'][-1],
        'final_imbalance': metrics['final_imbalance']
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

def thread_worker(row):
    """Worker pour exécuter la simulation dans un thread."""
    result = run_one_simulation(row)
    with stats_lock:
        global_stats.append(result)

def run_threaded(df, workers):
    """Exécution multi-threads des simulations."""
    global global_stats
    global_stats.clear()  # Réinitialiser avant chaque exécution

    num_threads = 4 if workers == "auto" else int(workers)
    num_threads = min(num_threads, len(df))  # éviter IndexError si CSV < 4 lignes
    threads = []

    start = time.perf_counter()
    for i in range(num_threads):
        t = threading.Thread(target=thread_worker, args=(df.iloc[i],))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    duration = time.perf_counter() - start
    return global_stats, duration

def plot_performance(seq_time, thr_time, out_dir):
    """Graphique comparant séq/thr,avec speedup."""
    speedup = seq_time / thr_time
    labels = ["Sequential", "Threaded"]
    times = [seq_time, thr_time]

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
    """Main function to run parallel parameter sweep using threading."""
    args = parse_args()
    df = pd.read_csv(args.params)
    args.out_dir.mkdir(exist_ok=True)

    seq_results, seq_time = run_sequential(df)
    print(f"Sequential time: {seq_time:.3f}s")

    thr_results, thr_time = run_threaded(df, args.workers)
    print(f"Threaded time: {thr_time:.3f}s")

    speedup = seq_time / thr_time
    print(f"Speedup: {speedup:.2f}x")
    print(
        "Note: le gain de performance dépend de la charge de calcul. "
        "Le speedup peut être limité par le GIL pour des tâches peu coûteuses.")

    pd.DataFrame(seq_results).to_csv(args.out_dir / "metrics_sequential.csv", index=False)
    pd.DataFrame(thr_results).to_csv(args.out_dir / "metrics_threaded.csv", index=False)
    plot_performance(seq_time, thr_time, args.out_dir)

if __name__ == "__main__":
    main()

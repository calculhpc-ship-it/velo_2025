import argparse
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI
from model import run_simulation

def parse_args():
    """Parse command line arguments for MPI parameter sweep."""
    parser = argparse.ArgumentParser(
        description="Bike simulation: distributed MPI version")
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()

def run_one_simulation(row):
    """Run one simulation and return compact aggregated metrics."""
    metrics = run_simulation(
        initial_mailly=int(row["init_mailly"]),
        initial_moulin=int(row["init_moulin"]),
        steps=int(row["steps"]),
        p1=float(row["p1"]),
        p2=float(row["p2"]),
        seed=int(row["seed"]),
    )
    result = {
        **row,
        "final_mailly": metrics["mailly"][-1],
        "final_moulin": metrics["moulin"][-1],
        "total_unmet_mailly": metrics["unmet_mailly"][-1],
        "total_unmet_moulin": metrics["unmet_moulin"][-1],
        "final_imbalance": metrics["final_imbalance"],
    }
    return result

def run_sequential(df):
    """Exécution séquentielle."""
    results = []
    start = time.perf_counter()
    for _, row in df.iterrows():
        results.append(run_one_simulation(row))
    duration = time.perf_counter() - start
    return results, duration

def main():
    """Main function to run parameter sweep using MPI."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()

    if rank == 0:

        # Master process
        args.out_dir.mkdir(exist_ok=True)

        df = pd.read_csv(args.params)
        tasks = df.to_dict(orient="records")

        seq_results, seq_time = run_sequential(df)
        print(f"Sequential execution time: {seq_time:.3f}s")

        #Distribution initiale des tâches aux workers
        task_index = 0
        for worker in range(1, min(size, len(tasks) + 1)):
            comm.send(tasks[task_index], dest=worker, tag=1)
            task_index += 1
        mp_results = []

        #Collecte dynamique des résultats
        start_mp = time.perf_counter()
        while task_index < len(tasks):
            status = MPI.Status()
            result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
            mp_results.append(result)

            worker = status.Get_source()
            comm.send(tasks[task_index], dest=worker, tag=1)
            task_index += 1

        #Récupérer les derniers résultats
        for worker in range(1, min(size, len(tasks) + 1)):
            result = comm.recv(source=worker, tag=2)
            mp_results.append(result)
        mp_duration = time.perf_counter() - start_mp

        #Envoyer signal d'arrêt
        for worker in range(1, size):
            comm.send(None, dest=worker, tag=0)

        #Affichage des temps et speedup
        print(f"MPI execution time: {mp_duration:.3f}s")
        speedup = seq_time / mp_duration
        print(f"Speedup (Sequential vs MPI): {speedup:.2f}x")
        print(
            "Note: MPI permet de répartir les tâches sur plusieurs nœuds, "
            "mais le speedup est limité par la latence réseau, la sérialisation "
            "des données et le master central.")

        #Sauvegarde des résultats
        pd.DataFrame(seq_results).to_csv(args.out_dir / "metrics_sequential.csv", index=False)
        pd.DataFrame(mp_results).to_csv(args.out_dir / "metrics_mpi.csv", index=False)
        labels = ["Sequential", "MPI"]
        times = [seq_time, mp_duration]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, times)
        plt.ylabel("Execution time (s)")
        plt.title("Sequential vs MPI performance")
        plt.text(
            0.5,
            max(times) * 0.9,
            f"Speedup = {speedup:.2f}x",
            ha="center",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7)
        )
        plt.tight_layout()
        plt.savefig(args.out_dir / "performance_comparison.png")
        plt.close()

    else:
        # Worker processes
        while True:
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
            if task is None:
                break
            result = run_one_simulation(task)
            comm.send(result, dest=0, tag=2)

if __name__ == "__main__":
    main()

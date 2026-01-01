import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from model import run_simulation

def parse_args():
    """Parse command line arguments for a single bike-sharing simulation."""
    parser = argparse.ArgumentParser(
        description="Single bike-sharing simulation")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--p1", type=float, required=True)
    parser.add_argument("--p2", type=float, required=True)
    parser.add_argument("--init-mailly", type=int, required=True)
    parser.add_argument("--init-moulin", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()

def main():
    """Run a single bike-sharing simulation and save results."""
    args = parse_args()
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Exécution de la simulation
    df, metrics = run_simulation(
        initial_mailly=args.init_mailly,
        initial_moulin=args.init_moulin,
        steps=args.steps,
        p1=args.p1,
        p2=args.p2,
        seed=args.seed,)
    # Sauvegarde série temporelle
    df.to_csv(out_path, index=False)
    # graphique
    if args.plot:
        plot_path = out_path.parent / "mailly.png"
        plt.figure(figsize=(10, 6))
        plt.plot(df["time"], df["mailly"], label="Mailly")
        plt.xlabel("Time")
        plt.ylabel("Number of bikes")
        plt.title("Bike-sharing simulation – Mailly")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
if __name__ == "__main__":
    main()

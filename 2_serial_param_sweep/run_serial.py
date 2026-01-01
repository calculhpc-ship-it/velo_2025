import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import run_simulation

def parse_args():
    """Parse command line arguments for serial parameter sweep."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--params", type=str, required=True) 
    parser.add_argument("--out-dir", type=str, required=True)  
    return parser.parse_args()

def main():
    """Main function to run serial parameter sweep."""
    args = parse_args()

    params_df = pd.read_csv(args.params)
    required_columns = ['steps', 'p1', 'p2', 'init_mailly', 'init_moulin', 'seed']
    for col in required_columns:
        if col not in params_df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    all_timeseries = []
    # Exécution des simulations
    for run_id, row in params_df.iterrows():
        timeseries_df, metrics = run_simulation(
            initial_mailly=row["init_mailly"],
            initial_moulin=row["init_moulin"],
            steps=row["steps"],
            p1=row["p1"],
            p2=row["p2"],
            seed=int(row["seed"]))
    
        metrics["run_id"] = run_id
        metrics["steps"] = row["steps"]
        metrics["p1"] = row["p1"]
        metrics["p2"] = row["p2"]
        metrics["init_mailly"] = row["init_mailly"]
        metrics["init_moulin"] = row["init_moulin"]
        metrics["seed"] = row["seed"]
        
        timeseries_df["run_id"] = run_id
        all_metrics.append(metrics)
        all_timeseries.append(timeseries_df)
    
    # Fusionne métriques - séries temp
    metrics_df = pd.DataFrame(all_metrics)
    timeseries_df = pd.concat(all_timeseries, ignore_index=True)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    # graphique
    generate_3plot(timeseries_df, metrics_df, out_dir)
def generate_3plot(timeseries_df, metrics_df, out_dir):
    """Generate the required 3-plot: mailly, moulin and balance for each simulation."""
    n_runs = len(metrics_df)
    
    fig, axes = plt.subplots(n_runs, 3, figsize=(15, 4 * n_runs))
    
    # Cas d'une seule simulation
    if n_runs == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_runs))
    
    for run_id in range(n_runs):
        run_data = timeseries_df[timeseries_df['run_id'] == run_id]
        run_metrics = metrics_df[metrics_df['run_id'] == run_id].iloc[0]
        
        # Mailly
        axes[run_id, 0].plot(run_data['time'], run_data['mailly'], 
                              color=colors[run_id], linewidth=2)
        axes[run_id, 0].axhline(y=run_metrics['init_mailly'], color='gray', linestyle='--', alpha=0.7)
        axes[run_id, 0].set_ylabel('Bikes at Mailly')
        axes[run_id, 0].set_title(f'Run {run_id}: Mailly Station')
        axes[run_id, 0].grid(True, alpha=0.3)
        
        # Moulin
        axes[run_id, 1].plot(run_data['time'], run_data['moulin'], 
                              color=colors[run_id], linewidth=2)
        axes[run_id, 1].axhline(y=run_metrics['init_moulin'], color='gray', linestyle='--', alpha=0.7)
        axes[run_id, 1].set_ylabel('Bikes at Moulin')
        axes[run_id, 1].set_title(f'Run {run_id}: Moulin Station')
        axes[run_id, 1].grid(True, alpha=0.3)
        
        # Balance
        balance = run_data['mailly'] - run_data['moulin']
        axes[run_id, 2].plot(run_data['time'], balance, color=colors[run_id], linewidth=2)
        axes[run_id, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[run_id, 2].axhline(y=run_metrics['final_imbalance'], color='red', linestyle='--', alpha=0.7,
                                label=f'Final: {run_metrics["final_imbalance"]}')
        axes[run_id, 2].set_xlabel('Time Step')
        axes[run_id, 2].set_ylabel('Balance (Mailly - Moulin)')
        axes[run_id, 2].set_title(f'Run {run_id}: Station Balance')
        axes[run_id, 2].legend(loc='upper right', fontsize='small')
        axes[run_id, 2].grid(True, alpha=0.3)
        
        # Infos paramètre
        param_text = f"p1={run_metrics['p1']}, p2={run_metrics['p2']}\n"
        param_text += f"seed={run_metrics['seed']}\n"
        param_text += f"unmet: Mly={run_metrics['unmet_mailly']}, Mln={run_metrics['unmet_moulin']}"
        axes[run_id, 2].text(0.02, 0.98, param_text, transform=axes[run_id, 2].transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                             fontsize=9)
    
    plt.suptitle('Parameter Sweep Results: Mailly, Moulin and Balance for Each Simulation', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_3plot.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()

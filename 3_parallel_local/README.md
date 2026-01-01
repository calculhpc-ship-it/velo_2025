# 3_parallel_local

Goal: accelerate the sweep using local CPU cores by using the three approaches.

- Threading
- Multiprocessing
- MPI

Run:

```bash
python run_threads.py --params params.csv --workers auto --out-dir thread/
python run_parallel.py --params params.csv --workers auto --out-dir multiprocessing/
python run_mpi.py --params params.csv --workers auto --out-dir mpi/
**Alternative utilisée en local:
mpirun -n 4 --oversubscribe python run_mpi.py --params params.csv --out-dir mpi/
```
Docoumentations
```
1-run_threadin.py
"Note: le gain de performance dépend de la charge de calcul.Le speedup peut être limité par le GIL pour des tâches peu coûteuses."
```
```
2-run_parallel.py
"Note: le multiprocessing évite le GIL mais introduit un surcoût lié à la création des processus et à la communication via Queue."
```
```
3-run_mpi.py
"Note: MPI permet de répartir les tâches sur plusieurs nœuds,mais le speedup est limité par la latence réseau, la sérialisation 
des données et le master central."
```


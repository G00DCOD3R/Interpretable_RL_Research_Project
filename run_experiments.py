import subprocess
from multiprocessing import Pool

def run_dagger(n):
    print(f"Starting dagger.py with n={n}")
    subprocess.run(["python3", "dagger.py", str(n)])
    print(f"Finished dagger.py with n={n}")

if __name__ == "__main__":
    ns = list(range(5, 100, 5))  
    with Pool(processes=len(ns)) as pool:
        pool.map(run_dagger, ns)

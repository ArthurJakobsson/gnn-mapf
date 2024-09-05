import subprocess

# multiprocessing libs
from multiprocessing import Pool

def run_sbatch(num_scens):
    main_command = "python -m master_process_runner 0 f t 100 1000 --num_parallel=50"
    main_command += f" --num_scens={num_scens}"
    subprocess.run(main_command.split(" "))

if __name__ == "__main__":

    scen_nums = [1,2,4,8,16,32,64,128]

    with Pool() as pool:
        results = pool.map(run_sbatch, scen_nums)

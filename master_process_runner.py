import os
import subprocess
import argparse
import pdb
import shutil
import multiprocessing
from datetime import datetime
import pickle

last_recorded_time = datetime.now()

class MyDict(dict):
   def __getitem__(self, item):
       return dict.__getitem__(self, item) % self

def get_iternum():
    global iternum
    return str(iternum)

def get_generate_initial():
    global generate_initial
    return generate_initial


def create_file_paths(args):
    file_path_master = MyDict({
        'data' : './data_collection/data',
        'logs' : '%(data)s/logs',
        'benchmark' : '%(data)s/benchmark_data',
        'mini_benchmark' : '%(data)s/mini_benchmark_data',
        'source_maps' : '%(mini_benchmark)s/maps' if args.mini_test else '%(benchmark)s/maps',
        'source_scens' : '%(mini_benchmark)s/scens' if args.mini_test else '%(benchmark)s/scens',
        'constant_npz' : '%(mini_benchmark)s/constant_npzs' if args.mini_test else '%(benchmark)s/constant_npzs',
        'initial_paths' : '%(mini_benchmark)s/initial_paths' if args.mini_test else '%(benchmark)s/initial_paths',
        # 'init_eecbs' : '%(initial_paths)s/eecbs_outputs',
        # 'init_eecbs_npz' : '%(initial_paths)s/eecbs_npzs',
        # 'init_processed' : '%(initial_paths)s/processed',
        # 'init_model' : '%(initial_paths)s/models',
        'experiment_logs' : '%(logs)s/EXP'+str(args.expnum),
        'iter_logs' : '%(experiment_logs)s/iter' + get_iternum(),
        'eecbs_outputs' : '%(initial_paths)s/eecbs_outputs' if get_generate_initial() else '%(iter_logs)s/eecbs_outputs',
        'eecbs_npzs': '%(initial_paths)s/eecbs_npzs' if get_generate_initial() else '%(iter_logs)s/eecbs_npzs',
        'processed' : '%(initial_paths)s/processed' if get_generate_initial() else '%(iter_logs)s/processed',
        'models' : '%(initial_paths)s/models' if get_generate_initial() else '%(iter_logs)s/models',
        'encountered_scens' : '%(iter_logs)s/encountered_scens',
        'timing' : '%(experiment_logs)s/timing_folder',
        'train_logs' : '%(logs)s/train_logs',
        'iternum' : get_iternum(),
        'meta_data' : '%(experiment_logs)s/meta_data.pickle'
    })  
    return file_path_master

def log_time(file_path_master, event_name):
    global last_recorded_time
    cur_time = datetime.now()
    with open(file_path_master["timing"]+"/master_timing.txt", mode='a') as file:
        file.write(f"{event_name} recorded at {cur_time}. \t\t Duration: \t {(cur_time-last_recorded_time).total_seconds()} \n")
    last_recorded_time  = cur_time

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expnum", help="experiment number", type=int)
    parser.add_argument('mini_test', type=lambda x: bool(str2bool(x)))
    parser.add_argument('generate_initial', type=lambda x: bool(str2bool(x)))
    parser.add_argument('num_samples', help="number of scens to simulate", type=int)
    parser.add_argument('max_samples', help="max number of scens to simulate", type=int)
    parser.add_argument('--num_parallel', type=int)
    args = parser.parse_args()
    expnum, mini_test, generate_initial, num_samples, max_samples = args.expnum, args.mini_test, args.generate_initial, args.num_samples, args.max_samples
    print(args.expnum)
    global iternum
    iternum = 0
    file_path_master = create_file_paths(args)

    pdb.set_trace()
    # if mini_test:
    #     source_maps_scens = "./data_collection/data/mini_benchmark_data"
    # else: 
    #     source_maps_scens = "./data_collection/data/benchmark_data"

    os.makedirs(file_path_master["experiment_logs"], exist_ok=True)
    os.makedirs(file_path_master["timing"], exist_ok=True)
    with open('file_path_master.pickle', 'wb') as fp:
        pickle.dump(file_path_master, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    num_cores = multiprocessing.cpu_count()
    first_iteration = "true"
    print("Current Path:", os.getcwd())

    # command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
    #                     f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
    #                     f"--num_parallel={args.num_parallel}", "--cutoffTime=20"])
    # print(command)
    command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={file_path_master['source_maps']}",  f"--scenFolder={file_path_master['source_scens']}", 
                            f"--outputFolder={file_path_master['eecbs_npzs']}", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
                            f"--num_parallel_runs={args.num_parallel}", "--cutoffTime=20"])
    print(command)
    # pdb.set_trace()
    
    log_time(file_path_master,"begin")
    
    if generate_initial:
        os.makedirs(file_path_master["initial_paths"], exist_ok=True)
        os.makedirs(file_path_master["eecbs_npzs"], exist_ok=True)
        os.makedirs(file_path_master["processed"], exist_ok=True)
        command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={file_path_master['source_maps']}",  f"--scenFolder={file_path_master['source_scens']}", 
                                f"--outputFolder={file_path_master['eecbs_npzs']}", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
                                f"--num_parallel_runs={args.num_parallel}", "--cutoffTime=20"])
        ### Example command for minibenchmark
        # python ./data_collection/eecbs_batchrunner2.py --mapFolder=./data_collection/data/mini_benchmark_data/maps 
        #           --scenFolder=./data_collection/data/mini_benchmark_data/scens --outputFolder=../data_collection/data/logs/EXP0/labels/raw/ 
        #           --expnum=0 --firstIter=true --iter=0 --num_parallel=50 --cutoffTime=20
        ### Example command for benchmark
        # python ./data_collection/eecbs_batchrunner2.py --mapFolder=./data_collection/data/benchmark_data/maps 
        #           --scenFolder=./data_collection/data/benchmark_data/scens --outputFolder=../data_collection/data/logs/EXP0/labels/raw/ 
        #           --expnum=0 --firstIter=true --iter=0 --num_parallel=50 --cutoffTime=20
        subprocess.run(command, shell=True, check=True)
        subprocess.run(["python", "./gnn/trainer.py", f"--experiment=exp{expnum}", f"--iternum={iternum}", f"--num_cores={num_cores}", f"--generate_initial={generate_initial}"])
        # print(command)
        # subprocess.run(["python", "./data_collection/eecbs_batchrunner.py", f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens", 
        #                 f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}"], check=True)
    
    else:
        assert (os.path.exists(f".{file_path_master['init_eecbs_npzs']}/train_warehouse_10_20_10_2_2_0.npz"))
    
    log_time(file_path_master,"initial generation")
    print("Done with first eecbs and first training run")
    # pdb.set_trace()
    
    generate_initial=False

    while True:        
        if not os.path.exists(f"./{LE}/iter{iternum}"):
            os.makedirs(f"./{LE}/iter{iternum}")

        # train the naive model
        subprocess.run(["python", "./gnn/trainer.py", f"--experiment=exp{expnum}", f"--iternum={iternum}", f"--num_cores={num_cores}", f"--generate_initial={generate_initial}"])
        log_time(file_path_master,f"training for iteration: {iternum}")
        # run cs-pibt new maps to create new scenes
        print((" ").join(["python", "./gnn/simulator.py", f"--exp_folder=./{LE}", f"--firstIter={first_iteration}",f"--source_maps_scens={source_maps_scens}", 
                        f"--iternum={iternum}", f"--num_samples={num_samples}", f"--max_samples={max_samples}",
                        f"--label_folder=./{LE}/labels/raw/"]))
        subprocess.run(["python", "./gnn/simulator.py", f"--exp_folder=./{LE}", f"--firstIter={first_iteration}",f"--source_maps_scens={source_maps_scens}", 
                        f"--iternum={iternum}", f"--num_samples={num_samples}", f"--max_samples={max_samples}",
                        f"--label_folder=./{LE}/labels/raw/"])
        first_iteration = "false"
        log_time(file_path_master,f"simulating for iteration: {iternum}")
        
        # feed failures into eecbs
        iternum+=1
        # Note: scen's should be taken from previous iteration but npz should be saved this next iteration number (since npzs have an extra)
        # subprocess.run(["python", "./data_collection/eecbs_batchrunner.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
        #                 f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}"]) # TODO figure out where eecbs is outputting files
        # command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
        #                 f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
        #                 f"--num_parallel={args.num_parallel}", "--cutoffTime=20"])
        subprocess.run(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
                        f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
                        f"--num_parallel_runs={args.num_parallel}", "--cutoffTime=20"]) # TODO figure out where eecbs is outputting files
        log_time(file_path_master,f"eecbs running for iteration: {iternum}")
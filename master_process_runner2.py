import subprocess
import argparse
import time

# run commands for data collection pipeline
# experiment with edge_attr(priorities), multistep input, multistep output

# example runs
'''
python master_process_runner2.py --data_path=$PROJECT/data/mini_benchmark_data \
    --exp_path=$PROJECT/data/logs/EXP_mini_priorities --iternum=0 \
    --mode=eecbs_batchrunner \
    --mode=constants_generator \
    --mode=dataloader \
    --clean
python master_process_runner2.py --data_path=$PROJECT/data/mini_benchmark_data \
    --exp_path=$PROJECT/data/logs/EXP_mini_priorities --iternum=0 \
    --mode=trainer \
    --models="SAGEConv" \
    --clean
'''

EDGE_ATTR_GNNS = ["ResGatedGraphConv", "GATv2Conv", "TransformerConv", "GENConv"]
NO_EDGE_ATTR_GNNS = ["SAGEConv"]

# args:
'''
    --data_path: benchmark data directory that contains maps/ and scens/. will contain:
        - constant_npzs: bd and goal npzs
    --exp_path: experiment results directory. will contain:
        - eecbs_outputs: path txts
        - eecbs_npzs: path npzs
        - processed: pt files
        - models: model pts
    --iternum
    
    --mode: <collect_data|train>
        - collect_data: constants_generator, eecbs_batchrunner, dataloader
        - train: trainer using models specified below
        
    --use_edge_attr
    --multistep_input
    --multistep_output  

    --clean: delete old files
'''


def eecbs_batchrunner(args):
    clean_batchrunner_cmd = f'rm -rf {args.exp_path}/iter{args.iternum}/'

    batchrunner_cmd = f'''python -m data_collection.eecbs_batchrunner5 --mapFolder={args.data_path}/maps \\
        --scenFolder={args.data_path}/scens \\
        --numAgents=50,100 \\
        --outputFolder={args.exp_path}/iter{args.iternum}/eecbs_outputs \\
        --num_parallel_runs={args.num_parallel_runs} \\
        "eecbs" \\
        --eecbsPath=./data_collection/eecbs/build_release4/eecbs \\
        --outputPathNpzFolder={args.exp_path}/iter{args.iternum}/eecbs_npzs \\
        --cutoffTime=5'''

    if args.clean:
        subprocess.run(clean_batchrunner_cmd, shell=True, check=True)

    t0 = time.time()
    subprocess.run(batchrunner_cmd, shell=True, check=True)
    print(f"eecbs_batchrunner: {time.strftime('%H:%M:%S',time.gmtime(time.time() - t0))}")


def constants_generator(args):
    clean_constants_exp_cmd = f'rm -rf {args.data_path}/logs/EXP_Collect_BD/'
    clean_constants_data_cmd = f'rm -rf {args.data_path}/constant_npzs/'

    constants_cmd = f'''python -m data_collection.constants_generator \\
        --mapFolder={args.data_path}/maps \\
        --scenFolder={args.data_path}/scens \\
        --constantMapAndBDFolder={args.data_path}/constant_npzs \\
        --outputFolder={args.data_path}/logs/EXP_Collect_BD/ \\
        --num_parallel_runs={args.num_parallel_runs} \\
        --deleteTextFiles=true \\
        "eecbs" \\
        --eecbsPath=./data_collection/eecbs/build_release4/eecbs \\
        --cutoffTime=1'''

    if args.clean:
        subprocess.run(clean_constants_exp_cmd, shell=True, check=True)
        subprocess.run(clean_constants_data_cmd, shell=True, check=True)
        
    t0 = time.time()
    subprocess.run(constants_cmd, shell=True, check=True)
    print(f"constants_generator: {time.strftime('%H:%M:%S',time.gmtime(time.time() - t0))}")
    

def dataloader(args):
    clean_pts_cmd = f'rm -rf {args.exp_path}/iter{args.iternum}/processed'
    clean_pts_csv_cmd = f'rm {args.exp_path}/iter{args.iternum}/status_data_processed.csv'
    
    dataloader_cmd = f'''python -m gnn.dataloader --mapNpzFile={args.data_path}/constant_npzs/all_maps.npz \\
        --bdNpzFolder={args.data_path}/constant_npzs \\
        --pathNpzFolder={args.exp_path}/iter{args.iternum}/eecbs_npzs \\
        --processedFolder={args.exp_path}/iter{args.iternum}/processed \\
        --k=5 \\
        --m=3 \\
        --num_priority_copies=10'''
    
    if args.clean:
        subprocess.run(clean_pts_cmd, shell=True, check=True)
        try: subprocess.run(clean_pts_csv_cmd, shell=True, check=True)
        except: pass
    
    t0 = time.time()
    subprocess.run(dataloader_cmd, shell=True, check=True)
    print(f"data_loader: {time.strftime('%H:%M:%S',time.gmtime(time.time() - t0))}")


def trainer(args, model):
    clean_models_cmd = f'rm -rf {args.exp_path}/models_{model}'

    trainer_cmd = f'''python -m gnn.trainer --exp_folder={args.exp_path} --experiment=exp0 --iternum={args.iternum} --num_cores=4 \\
        --processedFolders={args.exp_path}/iter{args.iternum}/processed \\
        --k=5 --m=3 --lr=0.01 \\
        --num_priority_copies=10 \\
        --gnn_name={model} \\
        --logging'''
    
    if args.use_edge_attr:
        trainer_cmd += '\\ \n--use_edge_attr'
    if args.multistep_input:
        trainer_cmd += '\\ \n--multistep_input'
    if args.multistep_output:
        trainer_cmd += '\\ \n--multistep_output'
    
    if args.clean:
        subprocess.run(clean_models_cmd, shell=True, check=True)
    
    t0 = time.time()
    subprocess.run(trainer_cmd, shell=True, check=True)
    print(f"trainer: {time.strftime('%H:%M:%S',time.gmtime(time.time() - t0))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--exp_path', type=str)
    parser.add_argument('--iternum', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--models', type=str)
    parser.add_argument('--use_edge_attr', action='store_true')
    parser.add_argument('--multistep_input', action='store_true')
    parser.add_argument('--multistep_output', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--num_parallel_runs', type=int, default=10)

    args = parser.parse_args()

    if args.mode == 'eecbs_batchrunner':
        eecbs_batchrunner(args)
    if args.mode == 'constants_generator':
        constants_generator(args)
    if args.mode == 'dataloader':
        dataloader(args)
    if args.mode == 'trainer':
        for model in args.models.strip().split():
            if args.use_edge_attr: assert(model in EDGE_ATTR_GNNS)
            trainer(args, model)
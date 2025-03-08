import subprocess
import argparse
import time

# run commands for data collection pipeline
# experiment with edge_attr(priorities), multistep input, multistep output

# example runs
'''
python sbatch_master_process_runner2.py --machine=omega \
    --collect_data \
    --data_path=mini_benchmark_data --exp_path=EXP_mini_0_1 \
    --clean
    
python sbatch_master_process_runner2.py --machine=omega \
    --trainer --models="ResGatedGraphConv" --use_edge_attr \
    --data_path=mini_benchmark_data --exp_path=EXP_mini_0_1 \
    --clean \
    --logging
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
    
    --mode: <collect_data|train>
        - collect_data: constants_generator, eecbs_batchrunner, dataloader
        - train: trainer
        
    --clean: delete old files
'''


def run_eecbs_batchrunner(args):
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
        try:
            subprocess.run(clean_batchrunner_cmd, shell=True, check=True)
        except: pass

    t0 = time.time()
    subprocess.run(batchrunner_cmd, shell=True, check=True)
    print(f"\neecbs_batchrunner: {time.strftime('%H:%M:%S',time.gmtime(time.time() - t0))}")


def run_constants_generator(args):
    clean_constants_exp_cmd = f'rm -rf {args.temp_bd_path}/'
    clean_constants_data_cmd = f'rm -rf {args.data_path}/constant_npzs/'

    constants_cmd = f'''python -m data_collection.constants_generator \\
        --mapFolder={args.data_path}/maps \\
        --scenFolder={args.data_path}/scens \\
        --constantMapAndBDFolder={args.data_path}/constant_npzs \\
        --outputFolder={args.temp_bd_path}/ \\
        --num_parallel_runs={args.num_parallel_runs} \\
        --deleteTextFiles=true \\
        "eecbs" \\
        --eecbsPath=./data_collection/eecbs/build_release4/eecbs \\
        --cutoffTime=1'''

    if args.clean:
        try: 
            subprocess.run(clean_constants_exp_cmd, shell=True, check=True)
            subprocess.run(clean_constants_data_cmd, shell=True, check=True)
        except: pass
        
    t0 = time.time()
    subprocess.run(constants_cmd, shell=True, check=True)
    print(f"\nconstants_generator: {time.strftime('%H:%M:%S',time.gmtime(time.time() - t0))}")
    

def run_dataloader(args):
    clean_pts_cmd = f'rm -rf {args.exp_path}/iter{args.iternum}/processed'
    clean_pts_csv_cmd = f'rm {args.exp_path}/iter{args.iternum}/status_data_processed.csv'
    
    dataloader_cmd = f'''python -m gnn.dataloader --mapNpzFile={args.data_path}/constant_npzs/all_maps.npz \\
        --bdNpzFolder={args.data_path}/constant_npzs \\
        --pathNpzFolder={args.exp_path}/iter{args.iternum}/eecbs_npzs \\
        --processedFolder={args.exp_path}/iter{args.iternum}/processed \\
        --k=5 \\
        --m=3 \\
        --num_priority_copies=10 \\
        --num_multi_inputs={args.num_multi_inputs} \\
        --num_multi_outputs={args.num_multi_outputs}'''
    
    if args.clean:
        try: 
            subprocess.run(clean_pts_cmd, shell=True, check=True)
            subprocess.run(clean_pts_csv_cmd, shell=True, check=True)
        except: pass
    
    t0 = time.time()
    subprocess.run(dataloader_cmd, shell=True, check=True)
    print(f"\ndata_loader: {time.strftime('%H:%M:%S',time.gmtime(time.time() - t0))}")


def run_trainer(args, model):
    clean_models_cmd = f'rm -rf {args.exp_path}/iter{args.iternum}/models_{model}'

    trainer_cmd = f'''python -m gnn.trainer --exp_folder={args.exp_path} --experiment=exp0 --iternum={args.iternum} --num_cores=4 \\
        --processedFolders={args.exp_path}/iter{args.iternum}/processed \\
        --k=5 --m=3 --lr=0.01 \\
        --num_priority_copies=10 \\
        --num_multi_inputs={args.num_multi_inputs} \\
        --num_multi_outputs={args.num_multi_outputs} \\
        --gnn_name="{model}"'''
    
    if args.logging:
        trainer_cmd += ' --logging'
    if args.use_edge_attr:
        trainer_cmd += ' --use_edge_attr'
    
    if args.clean:
        try:
            subprocess.run(clean_models_cmd, shell=True, check=True)
        except: pass
    
    t0 = time.time()
    subprocess.run(trainer_cmd, shell=True, check=True)
    print(f"\ntrainer: {time.strftime('%H:%M:%S',time.gmtime(time.time() - t0))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--machine', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--temp_bd_path', type=str, default='EXP_Collect_BD')
    parser.add_argument('--exp_path', type=str)
    parser.add_argument('--iternum', type=str, default='0')
    
    parser.add_argument('--collect_data', action='store_true')
    parser.add_argument('--eecbs_batchrunner', action='store_true')
    parser.add_argument('--constants_generator', action='store_true')
    parser.add_argument('--dataloader', action='store_true')
    parser.add_argument('--trainer', action='store_true')

    parser.add_argument('--models', type=str)
    parser.add_argument('--logging', action='store_true')

    parser.add_argument('--use_edge_attr', action='store_true')
    
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--num_parallel_runs', type=int, default=10)

    args = parser.parse_args()

    assert(args.machine in ['omega', 'psc'])
    if args.machine == 'omega':
        args.data_path = 'data_collection/data/' + args.data_path
        args.temp_bd_path = 'data_collection/data/' + args.temp_bd_path
        args.exp_path = 'data_collection/data/logs/' + args.exp_path
    elif args.machine == 'psc':
        args.data_path = '$PROJECT/data/' + args.data_path
        args.temp_bd_path = '$PROJECT/data/' + args.temp_bd_path
        args.exp_path = '$PROJECT/data/logs/' + args.exp_path

    if args.collect_data:
        args.eecbs_batchrunner = args.constants_generator = args.dataloader = True

    args.num_multi_inputs, args.num_multi_outputs = args.exp_path.strip().split('_')[-2:]
    print(f'multi input: {args.num_multi_inputs}, multi output: {args.num_multi_outputs}')

    if args.eecbs_batchrunner:
        run_eecbs_batchrunner(args)
    if args.constants_generator:
        assert(args.temp_bd_path != None)
        run_constants_generator(args)
    if args.dataloader:
        run_dataloader(args)
    if args.trainer:
        for model in args.models.strip().split():
            assert(model in EDGE_ATTR_GNNS or model in NO_EDGE_ATTR_GNNS)
            if args.use_edge_attr: assert(model in EDGE_ATTR_GNNS)
            run_trainer(args, model)
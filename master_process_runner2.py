import subprocess
import argparse

# run commands for data collection pipeline
# experiment with edge_attr(priorities), multistep input, multistep output

# example runs
'''
python master_process_runner2.py --data_path=medium_benchmark_data --exp_path=EXP_medium_priorities --iternum=0 \
    --mode=collect_data \
    --clean
python master_process_runner2.py --data_path=medium_benchmark_data --exp_path=EXP_medium_priorities --iternum=0 \
    --mode=train \
    --models="SAGEConv" \
    --clean
'''

EDGE_ATTR_GNNS = ["ResGatedGraphConv", "GATv2Conv", "TransformerConv", "GENConv"]
NO_EDGE_ATTR_GNNS = ["SAGEConv"]

# args:
'''
    --data_path: benchmark data directory in data_collection/data/ that contains maps/ and scens/. will contain:
        - constant_npzs: bd and goal npzs
    --exp_path: experiment results directory in data_collection/data/logs/. will contain:
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


def collect_data(args):
    # collect npzs
    clean_batchrunner_cmd = f'rm -rf {args.exp_path}/iter{args.iternum}/'

    batchrunner_cmd = f'''python -m data_collection.eecbs_batchrunner5 --mapFolder={args.data_path}/maps \
        --scenFolder={args.data_path}/scens \
        --numAgents=50,100 \
        --outputFolder={args.exp_path}/iter{args.iternum}/eecbs_outputs \
        --num_parallel_runs=10 \
        "eecbs" \
        --outputPathNpzFolder={args.exp_path}/iter{args.iternum}/eecbs_npzs \
        --firstIter=true --cutoffTime=5'''

    clean_constants_exp_cmd = f'rm -rf data_collection/data/logs/EXP_Collect_BD/'
    clean_constants_data_cmd = f'rm -rf {args.data_path}/constant_npzs/'

    constants_cmd = f'''python -m data_collection.constants_generator \
        --mapFolder={args.data_path}/maps \
        --scenFolder={args.data_path}/scens \
        --constantMapAndBDFolder={args.data_path}/constant_npzs \
        --outputFolder=data_collection/data/logs/EXP_Collect_BD/ \
        --num_parallel_runs=50 \
        --deleteTextFiles=true \
        "eecbs" \
        --firstIter=true --cutoffTime=1'''
    
    if args.clean:
        subprocess.run(clean_batchrunner_cmd, shell=True, check=True)
        subprocess.run(clean_constants_exp_cmd, shell=True, check=True)
        subprocess.run(clean_constants_data_cmd, shell=True, check=True)
    
    subprocess.run(batchrunner_cmd, shell=True, check=True)
    subprocess.run(constants_cmd, shell=True, check=True)
    
    # load pts
    clean_pts_cmd = f'rm -rf {args.exp_path}/iter{args.iternum}/processed'
    clean_pts_csv_cmd = f'rm {args.exp_path}/iter{args.iternum}/status_data_processed.csv'
    
    dataloader_cmd = f'''python -m gnn.dataloader --mapNpzFile={args.data_path}/constant_npzs/all_maps.npz \
        --bdNpzFolder={args.data_path}/constant_npzs \
        --pathNpzFolder={args.exp_path}/iter{args.iternum}/eecbs_npzs \
        --processedFolder={args.exp_path}/iter{args.iternum}/processed \
        --k=5 \
        --m=3 \
        --num_priority_copies=10'''
    
    if args.clean:
        subprocess.run(clean_pts_cmd, shell=True, check=True)
        try: subprocess.run(clean_pts_csv_cmd, shell=True, check=True)
        except: pass
    
    subprocess.run(dataloader_cmd, shell=True, check=True)


def train(args, model):
    clean_models_cmd = f'rm -rf {args.exp_path}/models_{model}'

    trainer_cmd = f'''python -m gnn.trainer --exp_folder={args.exp_path} --experiment=exp0 --iternum={args.iternum} --num_cores=4 \
        --processedFolders={args.exp_path}/iter{args.iternum}/processed \
        --k=5 --m=3 --lr=0.01 \
        --num_priority_copies=10 \
        --gnn_name={model} \
        --logging'''
    
    if args.use_edge_attr:
        trainer_cmd += '\ \n--use_edge_attr'
    if args.multistep_input:
        trainer_cmd += '\ \n--multistep_input'
    if args.multistep_output:
        trainer_cmd += '\ \n--multistep_output'
    
    if args.clean:
        subprocess.run(clean_models_cmd, shell=True, check=True)
    
    subprocess.run(trainer_cmd, shell=True, check=True)


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

    args = parser.parse_args()
    args.data_path = 'data_collection/data/'+args.data_path
    args.exp_path = 'data_collection/data/logs/'+args.exp_path

    if args.mode == 'collect_data':
        collect_data(args)
    elif args.mode == 'train':
        for model in args.models.strip().split():
            if args.use_edge_attr: assert(model in EDGE_ATTR_GNNS)
            train(args, model)
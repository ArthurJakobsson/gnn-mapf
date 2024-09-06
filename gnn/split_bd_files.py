import numpy as np
import argparse
import os
import shutil
import pdb
from multiprocessing import Pool

def split_bd_file(args):
    bd_file, bd_folder, num_partitions, completed_folder = args
    if bd_file == "all_maps.npz" or "part" in bd_file:
        return  # Skip if file is "all_maps.npz" or contains "part"
    
    bd_file_path = os.path.join(bd_folder, bd_file)
    bd_loaded = np.load(bd_file_path)
    keys = list(bd_loaded.keys())
    num_keys = len(keys)
    num_per_file = int(num_keys / num_partitions)
    
    for part_num in range(num_partitions):
        start_idx = part_num * num_per_file
        # Ensure last partition includes the remainder
        end_idx = (part_num + 1) * num_per_file if part_num != num_partitions - 1 else num_keys
        sub_keys = keys[start_idx:end_idx]
        
        # Create a sub-dictionary for this partition
        sub_dict = {key: bd_loaded[key] for key in sub_keys}
        
        # Save each partition as a new npz file
        output_file = os.path.join(bd_folder, f'{os.path.splitext(bd_file)[0]}_part_{part_num + 1}.npz')
        np.savez_compressed(output_file, **sub_dict)
        print(f'Saved {output_file}')
    
    completed_file_path = os.path.join(completed_folder, bd_file)
    shutil.move(bd_file_path, completed_file_path)
    print(f'Moved {bd_file} to {completed_file_path}')

def split_bd_parallel(bd_folder, num_partitions):
    bd_files_list = os.listdir(bd_folder)
    completed_folder = "data_collection/data/benchmark_data/completed_splitting"
    
    args = [(bd_file, bd_folder, num_partitions, completed_folder) for bd_file in bd_files_list]

    # Use multiprocessing Pool for parallel processing
    with Pool(processes=10) as pool:
        pool.map(split_bd_file, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bd_file_folder", help="bd files", type=str, required=True)
    parser.add_argument("--num_partition", help="number of splits", type=int, required=True)
    
    args = parser.parse_args()
    
    split_bd_parallel(args.bd_file_folder, args.num_partition)
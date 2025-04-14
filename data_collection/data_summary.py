# module load anaconda3/2022.10

import sys
import os
from glob import glob as glob
import csv

import numpy as np
import matplotlib.pyplot as plt

# column name: runtime or #high-level expanded
def get_txt_dicts(column):
    # eecbs csv files
    eecbs_outputs_path = f"{data_path}/eecbs_outputs"
    combined_csv_paths = glob(os.path.join(eecbs_outputs_path, '**', 'combined.csv'), recursive=True)
    
    txt_dict = {}

    for csv_path in combined_csv_paths:
        with open(csv_path, mode='r', newline='') as file:
            csv_reader = csv.DictReader(file)
            next(csv_reader)
            for row in csv_reader:
                if row:
                    if row['solution cost'] == -1: continue
                    f = f"{os.path.basename(row['agents']).split('.')[0]}.{row['agentNum']}.txt"
                    if column == 'runtime':
                        txt_dict[f] = float(row['runtime']) # map name to runtime
                    elif column == '#high-level expanded':
                        txt_dict[f] = float(row['#high-level expanded']) # map name to # high-level nodes expanded
                    else:
                        raise ValueError()

    all_counts = np.asarray([*txt_dict.values()])
    return all_counts, txt_dict

def plot_histogram(runtimes, data_path, title, x_label, filename, bins=20):
    plt.hist(runtimes, bins=bins, edgecolor='black')  # 'bins' defines how many bins you want
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.savefig(f"{data_path}/graphs/{filename}.png")
    plt.clf()

def cutoff_value(counts, cutoff):
    return cutoff, counts[counts >= cutoff]

def cutoff_percentile(counts, percentile=80):
    percentile_cutoff = np.percentile(counts, percentile)
    return percentile_cutoff, counts[counts >= percentile_cutoff]

def print_dict(s, d):
    print(s)
    sorted_items = sorted(d.items(), key=lambda kv: kv[1])
    print('\n'.join(f'{k}:{v}' for k, v in sorted_items[:5]))
    print('...')
    print('\n'.join(f'{k}:{v}' for k, v in sorted_items[-5:]))
    print()

'''
python data_collection/data_summary.py $PROJECT/data/logs/EXP_full/iter0/
python data_collection/data_summary.py $PROJECT/data/logs/EXP_new_maps/iter0/
'''

if __name__ == "__main__":
    args = sys.argv[1:]
    data_path = args[0]
    
    all_runtimes, csv_to_runtimes_dict = get_txt_dicts('runtime') # dict of dicts
    all_num_nodes, csv_to_num_nodes_dict = get_txt_dicts('#high-level expanded') 

    # threshold
    # runtime_cutoff, high_runtimes = cutoff_value(all_runtimes, 10)
    runtime_cutoff, high_runtimes = cutoff_percentile(all_runtimes, 50)
    # num_nodes_cutoff, high_num_nodes = cutoff_value(all_num_nodes, 10)
    num_nodes_cutoff, high_num_nodes = cutoff_percentile(all_num_nodes, 50)

    # plot and save histograms
    os.makedirs(f"{data_path}/graphs", exist_ok=True)
    plot_histogram(all_runtimes, data_path, 
                   'EECBS Runtimes', 
                   'Runtime', 
                   'all_runtimes', bins=50)
    plot_histogram(high_runtimes, data_path, 
                   f'EECBS Runtimes (minimum {runtime_cutoff:.3f} second runtime)', 
                   'Runtime', 
                   'high_runtimes', bins=50)
    print_dict('Runtime', csv_to_runtimes_dict)

    plot_histogram(all_num_nodes, data_path, 
                   'EECBS #High-level Nodes Expanded', 
                   '#High-level Nodes Expanded', 
                   'all_nodes', bins=50)
    plot_histogram(high_num_nodes, data_path, 
                   f'EECBS #High-level Nodes Expanded (minimum {num_nodes_cutoff:.3f} nodes)', 
                   '#High-level Nodes Expanded', 
                   'high_nodes', bins=50)
    print_dict('#High-level Nodes Expanded', csv_to_num_nodes_dict)
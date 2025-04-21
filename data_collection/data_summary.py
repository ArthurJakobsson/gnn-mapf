# module load anaconda3/2022.10

import sys
import os
from glob import glob as glob
import csv
import pandas as pd
import pdb

import numpy as np
import matplotlib.pyplot as plt

# column name: runtime or #high-level expanded
def get_txt_dicts(metric):
    # eecbs csv files
    eecbs_outputs_path = f"{data_path}/eecbs_outputs"
    combined_csv_paths = glob(os.path.join(eecbs_outputs_path, '**', 'combined.csv'), recursive=True)
    # print('csvs:', len(combined_csv_paths))
    
    txt_dict = {}

    for csv_path in combined_csv_paths:
        df = pd.read_csv(csv_path, skip_blank_lines=True)
        df = df[df['solution cost'] != -1]

        df['txt_name'] = df.apply(lambda row: f"{os.path.basename(row['agents']).split('.')[0]}.{row['agentNum']}.txt", axis=1)

        if metric == 'runtime':
            txt_dict.update(dict(zip(df['txt_name'], df['runtime'].astype(float))))
        elif metric == '#high-level expanded':
            txt_dict.update(dict(zip(df['txt_name'], df['#high-level expanded'].astype(float))))
        else:
            raise ValueError("Invalid metric")
        
    all_counts = np.asarray([*txt_dict.values()])

    return all_counts, txt_dict

def plot_histogram(runtimes, data_path, title, x_label, filename, bins=20):
    plt.hist(runtimes, bins=bins, edgecolor='black')  # 'bins' defines how many bins you want
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.savefig(f"{data_path}/graphs/{filename}")
    plt.clf()

def cutoff_value(counts, cutoff):
    return cutoff, counts[counts >= cutoff]

def cutoff_percentile(counts, percentile=80):
    percentile_cutoff = np.percentile(counts, percentile)
    return percentile_cutoff, counts[counts >= percentile_cutoff]

'''
python data_collection/data_summary.py $PROJECT/data/logs/EXP_full_increment/iter0/
python data_collection/data_summary.py $PROJECT/data/logs/EXP_full_increment_unfiltered/iter0/
python data_collection/data_summary.py $PROJECT/data/logs/EXP_new_maps/iter0/
'''

if __name__ == "__main__":
    args = sys.argv[1:]
    data_path = args[0]
    
    all_runtimes, txt_to_runtimes_dict = get_txt_dicts('runtime') # dict of dicts
    all_num_nodes, txt_to_num_nodes_dict = get_txt_dicts('#high-level expanded') 

    # threshold
    runtime_cutoff, high_runtimes = cutoff_value(all_runtimes, 10)
    # num_nodes_cutoff, high_num_nodes = cutoff_percentile(all_num_nodes, 50)

    # plot and save histograms
    os.makedirs(f"{data_path}/graphs", exist_ok=True)
    
    plot_histogram(all_runtimes, data_path, 
                   'EECBS Runtimes', 
                   'Runtime', 'all_runtimes.png', bins=50)
    plot_histogram(high_runtimes, data_path, 
                   f'EECBS Runtimes (minimum {runtime_cutoff:.3f} second runtime)', 
                   'Runtime', 'high_runtimes.png', bins=50)
    print('total:', len(all_runtimes))
    print('runtime >= 10s:', len(high_runtimes))

    plot_histogram(all_num_nodes, data_path, 
                   'EECBS #High-level Nodes Expanded', 
                   '#High-level Nodes Expanded', 'all_nodes.png', bins=50)
    # plot_histogram(high_num_nodes, data_path, 
    #                f'EECBS #High-level Nodes Expanded (minimum {num_nodes_cutoff:.3f} nodes)', 
    #                '#High-level Nodes Expanded', 'high_nodes.png', bins=50)
import os
import argparse
import pandas as pd
import numpy as np
import pdb

import matplotlib 
matplotlib.use('Agg')  # Required to not use display 
import matplotlib.pyplot as plt

def plot_bar_with_annotations(ax, data, title):
    """
    Input:
        ax: matplotlib axis
        data: grouped pandas DataFrame
        title: string
    """
    bars = data.plot(kind='bar', ax=ax)
    overall_mean = data.mean()
    ax.axhline(overall_mean, color='r', linestyle='--')
    ax.set_title(f"{title}: {overall_mean:.2f}")
    
    # Add numbers on top of each bar
    for bar in bars.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
                 ha='center', va='bottom')

def plotSingleEpoch(ax1, ax2, df, iter):
    """Input:
    """
    groupbyKeys = ['mapName', 'agentNum']
    
    # Plot success rate
    success_data = df.groupby(groupbyKeys)['success'].mean()
    plot_bar_with_annotations(ax1, success_data, f"Iter {iter}, Success")
    
    # Plot number of agents at goal
    agents_at_goal_data = df.groupby(groupbyKeys)['num_agents_at_goal'].mean()
    plot_bar_with_annotations(ax2, agents_at_goal_data, f"Iter {iter}, Agents at Goal")
    return success_data.mean(), agents_at_goal_data.mean()



def plotSummaryOverEpochs(main_folder):
    """Input:
        main_folder: string, path to the main folder containing the results of the training
        
    """

    # sorted_folders = sorted(os.listdir(main_folder))
    # for iter_folder in os.listdir(main_folder): # e.g. iter0
    df_per_iter = []
    for i in range(0, 100):
        iter_folder = os.path.join(main_folder, f"iter{i}")  # e.g. EXP_Test/iter0
        if os.path.exists(iter_folder):
            pymodel_outputs_folder = os.path.join(main_folder, iter_folder, "pymodel_outputs")
            if not os.path.exists(pymodel_outputs_folder):
                continue

            df_list = []
            for mapfolder in os.listdir(pymodel_outputs_folder):
                csv_file = os.path.join(pymodel_outputs_folder, mapfolder, "csvs/combined.csv")
                if os.path.exists(csv_file):
                    # Load the results
                    results = pd.read_csv(csv_file)
                    df_list.append(results)

            df = pd.concat(df_list)
            df_per_iter.append(df)
    
    num_iters = len(df_per_iter)
    ncols = 2
    nrows = num_iters #int(np.ceil(num_iters / ncols))
    print(ncols, nrows)
    fig = plt.figure(figsize=(ncols*6+4, nrows*6))

    for i, df in enumerate(df_per_iter):
        ax1 = fig.add_subplot(nrows, ncols, 2*i+1)
        ax2 = fig.add_subplot(nrows, ncols, 2*i+2)

        overall_success_mean, overall_agents_at_goal_mean = plotSingleEpoch(ax1, ax2, df, i)

    plt.tight_layout()
    plt.savefig(os.path.join(main_folder, "results_per_epoch.png"))
    plt.close()

    ### Plot overall success rate and agents at goal over epochs
    overall_success_mean_list = [df['success'].mean() for df in df_per_iter]
    overall_agents_at_goal_mean_list = [df['num_agents_at_goal'].mean() for df in df_per_iter]

    fig, ax1 = plt.subplots(figsize=(2*6+4, 6))
    ax2 = ax1.twinx() # This plots the second y axis on the same plot

    ax1.plot(range(num_iters), overall_success_mean_list, 'b--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Overall Success Rate', color='b')
    ax1.tick_params('y', colors='b')

    ax2.plot(range(num_iters), overall_agents_at_goal_mean_list, 'r-.')
    ax2.set_ylabel('Overall Agents at Goal', color='r')
    ax2.tick_params('y', colors='r')

    plt.title('Overall Success Rate and Agents at Goal over Epochs')
    plt.savefig(os.path.join(main_folder, "results_summary.png"))
    plt.close()

### Example call
"""
python plot_results.py --main_folder=/home/rishi/research/gnn-mapf/data_collection/data/logs/EXP_den312d_test
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type=str)
    args = parser.parse_args()

    plotSummaryOverEpochs(args.main_folder)
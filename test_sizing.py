import os
import torch
import time

def load_graphs(folder_path, num_graphs=200):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])[:num_graphs]
    graphs = [torch.load(os.path.join(folder_path, f)) for f in files]
    return graphs

def save_arrays(graphs, sizes, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    arrays = []
    for size in sizes:
        arrays.append(graphs[:size])
        torch.save(arrays[-1], os.path.join(output_folder, f'graphs_{size}.pt'))

def time_loading_and_access(output_folder, sizes):
    timings = {}
    for size in sizes:
        start_time = time.time()
        loaded_array = torch.load(os.path.join(output_folder, f'graphs_{size}.pt'))
        _ = loaded_array[0]
        end_time = time.time()
        timings[size] = end_time - start_time
    return timings

def main():
    folder_path = '/home/arthur/snap/snapd-desktop-integration/current/Documents/gnn-mapf/data_collection/data/logs/EXP_one_over/iter0/processed'  # Replace with the folder path containing your .pt files
    output_folder = 'repackaged_test'
    sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    graphs = load_graphs(folder_path)
    save_arrays(graphs, sizes, output_folder)
    timings = time_loading_and_access(output_folder, sizes)
    for size, timing in timings.items():
        print(f'Size {size}: {timing:.6f} seconds')

if __name__ == "__main__":
    main()
import os
import shutil
import sys; args = sys.argv[1:]
import re

'''
python data_collection/rename_maps_scens.py $PROJECT/data/benchmark_data
python data_collection/rename_maps_scens.py data_collection/data/benchmark_data

'''

data_path = args[0]

assert('maps' in os.listdir(data_path) and 'scens' in os.listdir(data_path))
map_path = os.path.join(data_path, 'maps')
scen_path = os.path.join(data_path, 'scens')
assert(os.path.isdir(map_path) and os.path.isdir(scen_path))

new_data_path = os.path.join(os.path.dirname(data_path), os.path.basename(data_path) + '_copy')
print('New data path:', new_data_path)
new_map_path = os.path.join(new_data_path, 'maps')
new_scen_path = os.path.join(new_data_path, 'scens')
os.makedirs(new_map_path, exist_ok=True)
os.makedirs(new_scen_path, exist_ok=True)

for filename in os.listdir(scen_path):
    pattern = r'-random-\d+\.scen$'
    match = re.search(pattern, filename)
    assert(match != None)
    new_filename = filename[:-len(match.group())].replace('-', '_') + match.group()
    
    src_file = os.path.join(scen_path, filename)
    dst_file = os.path.join(new_scen_path, new_filename)
    shutil.copy(src_file, dst_file)

for filename in os.listdir(map_path):
    if '-' in filename:
        new_filename = filename.replace('-', '_')
    else:
        new_filename = filename
        
    src_file = os.path.join(map_path, filename)
    dst_file = os.path.join(new_map_path, new_filename)
    shutil.copy(src_file, dst_file)

assert(len(os.listdir(map_path)) == len(os.listdir(new_map_path)))
assert(len(os.listdir(scen_path)) == len(os.listdir(new_scen_path)))

print('Done.')
    


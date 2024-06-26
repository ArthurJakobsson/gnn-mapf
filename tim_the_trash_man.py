#        ________________   ___/-\___     ___/-\___     ___/-\___
#      / /             ||  |---------|   |---------|   |---------|
#     / /              ||   |       |     | | | | |     |   |   |
#    / /             __||   |       |     | | | | |     | | | | |
#   / /   \\        I  ||   |       |     | | | | |     | | | | |
#  (-------------------||   | | | | |     | | | | |     | | | | |
#  ||               == ||   |_______|     |_______|     |_______|
#  ||   ACME TRASH CO  | =============================================
#  ||          ____    |                                ____      |
# ( | o      / ____ \                                 / ____ \    |)
#  ||      / / . . \ \                              / / . . \ \   |
# [ |_____| | .   . | |____________________________| | .   . | |__]
#           | .   . |                                | .   . |
#            \_____/                                  \_____/
#  _                 _     
# | |               | |    
# | |_ _ __ __ _ ___| |__  
# | __| '__/ _` / __| '_ \ 
# | |_| | | (_| \__ \ | | |
#  \__|_|  \__,_|___/_| |_|

import os
import shutil

# clean root

dir_name = "./"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".scen"):
        os.remove(os.path.join(dir_name, item))


# clean raw_data

dir_name = "./data_collection/eecbs/raw_data/bd"
bd_files = os.listdir(dir_name)
for item in bd_files:
    os.remove(os.path.join(dir_name, item))

dir_name = "./data_collection/eecbs/raw_data/paths"
path_files = os.listdir(dir_name)
for item in path_files:
    os.remove(os.path.join(dir_name, item))

dir_name = "./data_collection/eecbs/raw_data/"
eecbs_files = os.listdir(dir_name)
for item in eecbs_files:
    if (item != "bd") and (item != "paths"):
        shutil.rmtree(os.path.join(dir_name, item))

# clean logs
shutil.rmtree("./data_collection/data/logs/")
os.mkdir("./data_collection/data/logs/")
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
#    __     _                       __     __                              
#   / /_   (_)   ____ ___          / /_   / /_   ___                       
#  / __/  / /   / __ `__ \        / __/  / __ \ / _ \                      
# / /_   / /   / / / / / /       / /_   / / / //  __/                      
# \__/  /_/   /_/ /_/ /_/        \__/  /_/ /_/ \___/                       
                                                                         
#    __                             __                                     
#   / /_   _____  ____ _   _____   / /_           ____ ___   ____ _   ____ 
#  / __/  / ___/ / __ `/  / ___/  / __ \         / __ `__ \ / __ `/  / __ \
# / /_   / /    / /_/ /  (__  )  / / / /        / / / / / // /_/ /  / / / /
# \__/  /_/     \__,_/  /____/  /_/ /_/        /_/ /_/ /_/ \__,_/  /_/ /_/ 

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
if os.path.exists(dir_name):
    bd_files = os.listdir(dir_name)
    for item in bd_files:
        os.remove(os.path.join(dir_name, item))

dir_name = "./data_collection/eecbs/raw_data/paths"
if os.path.exists(dir_name):
    path_files = os.listdir(dir_name)
    for item in path_files:
        os.remove(os.path.join(dir_name, item))

dir_name = "./data_collection/eecbs/raw_data/"
if os.path.exists(dir_name):
    eecbs_files = os.listdir(dir_name)
    for item in eecbs_files:
        if (item != "bd") and (item != "paths") and (item != ".DS_Store"):
            shutil.rmtree(os.path.join(dir_name, item))

os.makedirs(dir_name+"/bd", exist_ok=True)
os.makedirs(dir_name+"/paths", exist_ok=True)

# clean logs
shutil.rmtree("./data_collection/data/logs/")
os.mkdir("./data_collection/data/logs/")
os.mkdir("./data_collection/data/logs/train_logs")

shutil.rmtree("./timing_folder/")
os.mkdir("./timing_folder/")

print("Tim has collected your trash 😘")
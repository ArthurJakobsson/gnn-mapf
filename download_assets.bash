mkdir data
cd data

# Download maps
wget https://movingai.com/benchmarks/mapf/mapf-map.zip
unzip mapf-map.zip -d mapf-map

# Download random scens; note other scens are available to try too
wget https://movingai.com/benchmarks/mapf/mapf-scen-random.zip
unzip mapf-scen-random.zip && mv scen-random mapf-scen-random  # Rename unzipped folder for consistency

# Download backward dijkstras npzs
gdown https://drive.google.com/drive/folders/1X79K-GRcZn9YvdwLmI3yCdcmRSeYIm_K?usp=drive_link -O constant_npzs/ --folder

# Download all_maps npzs
gdown https://drive.google.com/uc?id=1HIB7FvJUjxVzQyByHlWrJCdUiI7GrCbj -O constant_npzs/

# Download model
mkdir model
gdown https://drive.google.com/uc?id=1-1_onJLmRp5wxp0I6CKqV1bZmc25qcsM -O model/

# Create logs folder
cd ..
mkdir logs  # Optional, recommended for consistency with batch_runner.py
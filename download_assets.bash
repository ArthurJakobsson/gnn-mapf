if [ ! -d "data" ]; then
    mkdir data
fi
cd data

# Download maps
if [ ! -d "mapf-map" ]; then
    wget https://movingai.com/benchmarks/mapf/mapf-map.zip
    unzip mapf-map.zip -d mapf-map
fi

# Download random scens; note other scens are available to try too
if [ ! -d "mapf-scen-random" ]; then
    wget https://movingai.com/benchmarks/mapf/mapf-scen-random.zip
    unzip mapf-scen-random.zip -d mapf-scen-random # Rename unzipped folder for consistency
fi

# Download backward dijkstras npzs
gdown https://drive.google.com/drive/folders/1X79K-GRcZn9YvdwLmI3yCdcmRSeYIm_K?usp=drive_link -O constant_npzs/ --folder --continue

# Download all_maps npzs
gdown https://drive.google.com/uc?id=1HIB7FvJUjxVzQyByHlWrJCdUiI7GrCbj -O constant_npzs/ --continue

# Download model
if [ ! -d "model" ]; then
    mkdir model
fi
gdown https://drive.google.com/uc?id=1-1_onJLmRp5wxp0I6CKqV1bZmc25qcsM -O model/ --continue

# Create logs folder
cd ..
if [ ! -d "logs" ]; then
    mkdir logs  # Optional, recommended for consistency with batch_runner.py
fi
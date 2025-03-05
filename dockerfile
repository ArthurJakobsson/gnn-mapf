FROM nvcr.io/nvidia/isaac-sim:4.1.0
RUN apt update
RUN apt install -y wget htop git

# Clean up the apt cache to reduce the image size
RUN rm -rf /var/lib/apt/lists/*
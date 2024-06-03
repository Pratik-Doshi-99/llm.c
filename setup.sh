echo "________________________________________________________________________"
echo "Updting apt next..."
echo "________________________________________________________________________"
apt update

echo "________________________________________________________________________"
echo "Apt Updated"
echo "________________________________________________________________________"
echo "Installing vim next..."
echo "________________________________________________________________________"
printf "Y" | apt install vim

echo "________________________________________________________________________"
echo "Vim Installed"
echo "________________________________________________________________________"
echo "Installing net-tools next..."
echo "________________________________________________________________________"
apt printf "Y" | install net-tools

# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

# pytorch nightly (optional) https://pytorch.org/get-started/locally/
# conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia

# pip installs so we can tokenize the FineWeb dataset
yes | pip install tqdm tiktoken requests datasets

# install cudnn so we can use FlashAttention and run fast (optional)
# https://developer.nvidia.com/cudnn-downloads
# for me, CUDA 12 (run `nvcc --version`) running on Linux x86_64 Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install libcudnn9-dev-cuda-12

# "install" cudnn-frontend to ~/
git clone https://github.com/NVIDIA/cudnn-frontend.git

# install MPI (optional, if you intend to use multiple GPUs)
apt install openmpi-bin openmpi-doc libopenmpi-dev

# tokenize the FineWeb dataset 10B tokens sample (takes ~1 hour, get lunch?)
# writes ~19GB of raw GPT-2 tokens to dev/data/fineweb10B
# and ~46GB in ~/.cache/huggingface/datasets/HuggingFaceFW___fineweb
#git clone https://github.com/karpathy/llm.c.git
cd llm.c
#python dev/data/fineweb.py --version 10B

# compile llm.c (mixed precision, with cuDNN flash-attention)
# first compilation is ~1 minute, mostly due to cuDNN
make train_gpt2cu USE_CUDNN=1

# train on a single GPU
# ./train_gpt2cu \
#     -i "dev/data/fineweb10B/fineweb_train_*.bin" \
#     -j "dev/data/fineweb10B/fineweb_val_*.bin" \
#     -o log124M \
#     -e "d12" \
#     -b 64 -t 1024 \
#     -d 524288 \
#     -r 1 \
#     -z 1 \
#     -c 0.1 \
#     -l 0.0006 \
#     -q 0.0 \
#     -u 700 \
#     -n 5000 \
#     -v 250 -s 20000 \
#     -h 1

# if you have multiple GPUs (e.g. 8), simply prepend the mpi command, e.g.:
# mpirun -np 8 ./train_gpt2cu \ ... (the rest of the args are same)
# mpirun -np 2 ./train_gpt2cu \ 
#     -i "dev/data/fineweb10B/fineweb_train_*.bin" \
#     -j "dev/data/fineweb10B/fineweb_val_*.bin" \
#     -o log124M \
#     -e "d12" \
#     -b 32 -t 1024 \
#     -d 524288 \
#     -r 1 \
#     -z 1 \
#     -c 0.1 \
#     -l 0.0006 \
#     -q 0.0 \
#     -u 700 \
#     -n 5000 \
#     -v 250 -s 20000 \
#     -h 1
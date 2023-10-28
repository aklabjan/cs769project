conda create --name goemotion_env
conda activate goemotion_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install transformers -c conda-forge
conda install attrdict -c conda-forge

# for "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
# please check your cuda version with "nvcc -V", it is very important. And then according to your cuda version to install PyTorch referring the cmd on this page "https://pytorch.org/get-started/locally/"
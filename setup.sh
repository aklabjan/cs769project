conda create --name --y goemotion_env python=3.8.18
conda activate goemotion_env
conda install --y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install --y  transformers -c conda-forge
conda install --y attrdict -c conda-forge
pip install sklearn==0.0
# for "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
# please check your cuda version with "nvcc -V", it is very important. And then according to your cuda version to install PyTorch referring the cmd on this page "https://pytorch.org/get-started/locally/"

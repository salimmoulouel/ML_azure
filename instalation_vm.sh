#chmod 400 file_emp
#ssh i- file_pem azureuser@publicipadress



lspci | grep -i NVIDIA
CUDA_REPO_PKG=cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

wget -O /tmp/${CUDA_REPO_PKG} https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/${CUDA_REPO_PKG} 

sudo dpkg -i /tmp/${CUDA_REPO_PKG}

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub 

rm -f /tmp/${CUDA_REPO_PKG}

sudo apt-get update

sudo apt-get install cuda-drivers


#build tunelling port between distant machine and local
#ssh -i key_pem -L localport:localhost:distantport azureuser@publicipadress 


#Install CUDA 10
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y cuda=10.0.130-1
rm cuda-repo-ubuntu1804_*_amd64.deb


#Install cuDNN 7.6.5
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
sudo apt-get install -y libcudnn7=7.6.5.32-1+cuda10.0
sudo apt-get install -y libcudnn7-dev=7.6.5.32-1+cuda10.0
rm nvidia-machine-learning-repo-ubuntu1804_*_amd64.deb
sudo ldconfig


#Environment variables
export LD_LIBRARY_PATH=/usr/local/cuda-10.0:${LD_LIBRARY_PATH}



#Install TensorFlow
sudo apt-get install -y python3-dev python3-pip
sudo pip3 install tensorflow-gpu




python3
>>> import tensorflow as tf
>>> session = tf.Session()

# A self-contained order taking robot
_beep boop!_
- Linux [have not tested on WSL2 / windows]
- ollama, 
- vosk for speech detection 
- transformer models from huggingface for text to speech. 

## OLLAMA
[Download Ollama Here](https://ollama.com/download)

- Linux: `curl -fsSL https://ollama.com/install.sh | sh`

- Some times ollama will barf
- Issue Reference: [https://github.com/ollama/ollama/issues/8638]
```
sudo systemctl stop ollama
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
sudo systemctl start ollama

```

### CUDA // LINUX
[https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu]
```
export KEYRING=cuda-keyring_1.1-1_all.deb
export DISTRO=ubuntu2404
export ARCH=x86_64
wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$ARCH/$KEYRING
sudo dpkg -i $KEYRING
sudo apt-get update
sudo apt-get install cuda-toolkit
``` 

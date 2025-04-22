sudo systemctl stop ollama
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
sudo systemctl start ollama

https://github.com/ollama/ollama/issues/8638

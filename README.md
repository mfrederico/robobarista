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

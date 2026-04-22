#!/bin/sh

# install Ollama and download the Qwen3.5:27b model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:27b

# install anaconda to create conda python environment
curl -fsSL https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh | sh
conda create -n tiny-agent python=3.11.7
conda activate tiny-agent
cd ~
git clone https://github.com/ninevoltz/tiny-agent.git
cd tiny-agent
pip install -r requirements.txt

python tiny-agent.py

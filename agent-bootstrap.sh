#!/bin/sh

# Ask user if they want to install Ollama
echo "Would you like to install Ollama and download the Qwen3.5:27b model? (yes/no)"
read -r response

# Convert to lowercase for case-insensitive comparison
response_lower=

if [ "" = "yes" ] || [ "" = "y" ]; then
    # install Ollama and download the Qwen3.5:27b model
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Downloading Qwen3.5:27b model..."
    ollama pull qwen3.5:27b
else
    echo "Skipping Ollama installation."
fi

# Ask user if they want to install SearXNG
echo "Would you like to install SearXNG to enable web search for your agent? (yes/no)"
read -r response

# Convert to lowercase for case-insensitive comparison
response_lower=

if [ "" = "yes" ] || [ "" = "y" ]; then
    echo "Installing SearXNG..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Skipping SearXNG installation."
fi

# update the system and install python3
sudo apt update
sudo apt install python3-full git

# create python virtual environment
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python tiny-agent.py

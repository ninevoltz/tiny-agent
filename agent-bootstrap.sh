#!/bin/sh

# update the system and install python3
sudo apt update
sudo apt install python3-full git yq

# create python virtual environment
rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Ask user if they want to install Ollama
echo "Would you like to install Ollama and download the Qwen3.5:27b model? (yes/no)"
read -r response

if [ "$response" = "yes" ] || [ "$response" = "y" ]; then
    # install Ollama and download the Qwen3.5:27b model
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Downloading Qwen3.5:27b model..."
    ollama pull qwen3.6:35b

    SERVICE="ollama"
    OVERRIDE_DIR="/etc/systemd/system/${SERVICE}.service.d"
    OVERRIDE_FILE="${OVERRIDE_DIR}/override.conf"

    echo "📁 Creating systemd drop-in directory..."
    sudo mkdir -p "$OVERRIDE_DIR"

    echo "✍️  Writing environment variables to override file..."
    sudo tee "$OVERRIDE_FILE" > /dev/null << 'EOF'
    [Service]
        Environment="OLLAMA_HOST=0.0.0.0:11434"
        Environment="OLLAMA_CONTEXT_LENGTH=262144"
EOF

    echo "Reloading systemd daemon..."
    sudo systemctl daemon-reload

    echo "Restarting ${SERVICE} service..."
    sudo systemctl restart "${SERVICE}"

    echo "Verifying applied environment variables:"
        systemctl show "${SERVICE}" --property=Environment
else
    echo "Skipping Ollama installation."
fi

# Ask user if they want to install SearXNG
echo "Would you like to install SearXNG to enable web search for your agent? (yes/no)"
read -r response

if [ "$response" = "yes" ] || [ "$response" = "y" ]; then
    echo "Installing SearXNG..."
    
    # Clean up existing directory if any
    if [ -d "searxng" ]; then
        rm -rf searxng
    fi

    # Clone repository
    git clone https://github.com/searxng/searxng.git searxng
    cd searxng

    pip install -r requirements.txt
    
    # Configure html and JSON
    yq -iy '.search.formats = ["html", "json"] | del(.use_default_settings)' searx/settings.yml

    # Start SearXNG in the background
    nohup ./manage webapp.run > searxng.log 2>&1 &
    echo "SearXNG is running on http://localhost:8888"
    
    # Go back to the main directory
    cd ..
else
    echo "Skipping SearXNG installation."
fi

python tiny-agent.py

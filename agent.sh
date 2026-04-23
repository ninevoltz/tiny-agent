#!/bin/sh
PROJECT_PATH=~/git-projects/tiny-agent

cd ${PROJECT_PATH}
. .venv/bin/activate
cd searxng
nohup ./manage webapp.run > searxng.log 2>&1 &
cd ..
python tiny-agent.py --base-url  http://192.168.3.74:11434 --model qwen3.6:27b

killall searxng

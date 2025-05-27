#!/bin/bash


current_dir="$PWD"
echo "Current working directory: $current_dir"

python3 -m venv venv

# python3 -m venv venv
chmod 777 "$current_dir/venv/bin/activate"
source "$current_dir/venv/bin/activate"

pip3 install --upgrade pip
pip3 install -r requirements.txt

# To open jupyter lab in localhost
# jupyter lab --allow-root --ip 0.0.0.0 --port 9999

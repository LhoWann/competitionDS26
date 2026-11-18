#!/bin/bash
# One-time Google login for the Colab CLI inside WSL (Application Default Credentials with the scopes Colab needs).
# Run it yourself in a terminal: wsl -d Ubuntu-24.04 -- bash /mnt/d/DataSlayer26/colab/login.sh
set -e
export PATH=$HOME/.local/bin:$PATH
gcloud auth application-default login --no-launch-browser \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab --auth=adc sessions && echo "LOGIN OK"
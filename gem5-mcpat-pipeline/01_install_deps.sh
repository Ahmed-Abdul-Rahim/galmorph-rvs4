#!/usr/bin/env bash
# Run this once in your WSL Ubuntu shell, with sudo rights.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
    build-essential git m4 scons zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev \
    libgoogle-perftools-dev python3 python3-dev python3-pip \
    libboost-all-dev pkg-config libhdf5-serial-dev \
    python3-pydot doxygen

echo "Dependencies installed. Next: run 02_build_gem5.sh"

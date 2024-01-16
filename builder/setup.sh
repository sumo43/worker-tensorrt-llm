#!/bin/bash

# NOTE: This script is not ran by default for the template docker image.
#       If you use a custom base image you can add your required system dependencies here.

# install triton with example
git clone https://github.com/triton-inference-server/python_backend -b r23.11
cd python_backend
mkdir -p models/add_sub/1/
cp examples/add_sub/model.py models/add_sub/1/model.py
cp examples/add_sub/config.pbtxt models/add_sub/config.pbtxt
cd /

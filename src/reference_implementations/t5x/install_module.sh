#!/bin/bash

mkdir t5x_git_repo_path
cd t5x_git_repo_path
git clone --branch=main https://github.com/google-research/t5x
cd t5x
pip install .
cd ..
cd ..
rm -r -f t5x_git_repo_path

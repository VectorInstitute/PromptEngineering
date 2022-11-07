#!/bin/bash

mkdir prompt_tuning_git_repo_path
cd prompt_tuning_git_repo_path
git clone --branch=main https://github.com/google-research/prompt-tuning
cd prompt-tuning
pip install .
cd ..
cd ..
rm -r -f prompt_tuning_git_repo_path

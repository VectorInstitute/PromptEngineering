#!/bin/bash

brew install python@3.8
python3.8 -m venv env
source env/bin/activate
./env/bin/python3.8 -m pip install --upgrade pip
./env/bin/python3.8 -m pip install -e .[dev]

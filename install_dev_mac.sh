#!/bin/bash

brew install python@3.9
python3.9 -m venv env
source env/bin/activate
./env/bin/python3.9 -m pip install --upgrade pip
./env/bin/python3.9 -m pip install -e .[dev]

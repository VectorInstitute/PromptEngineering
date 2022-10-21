#!/bin/bash

version=3.8.15

# Install pyenv
curl https://pyenv.run | bash

# Follow the instruction to modify ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# Install the latest Python from source code
pyenv install ${version} 

# Check installed Python versions
pyenv versions

# Switch Python version
pyenv global ${version}

# Check where Python is actually installed
pyenv prefix

# Check the current Python version
python -V

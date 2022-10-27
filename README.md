# Prompt Engineering
This repository holds all of the code associated with the project considering prompt engineering for large language models. This includes work around reference implementations, demo notebooks, and fairness and bias evaluation.

The static code checker runs on python3.9

# Installing dependencies
Remember to activate your associated virtual environment in order to install the dependencies in a separate env from your machine.
```
pip install --upgrade pip
pip install -r requirements.txt
```
## For Developers and Contributers
If you wish to install the package in the editable mode with all the development requirements, you should use the following command once you activate your virtual environment:
```
pip install --upgrade pip
pip install -e .[dev]
```

On vector's cluster (Mars), you can alternatively run the following script to set up the development environment with the necessary `python3.9`:
```
bash install_dev_cluster.sh
```

Similarly on a local mac, you can simply run the following script to set up `python3.9` and the env:
```
bash install_dev_mac.sh
```

### Using Pre-commit Hooks
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```

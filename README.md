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

On vector's cluster, you must run the following script to set up the development environment with the necessary `python3.9`:
```
bash install_dev_cluster.sh
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
# Running T5x
We currently support running T5x only on the vector's cluster.
Make sure you install the cluster dependencies via the following command:
```
bash install_dev_cluster.sh
```

To start training a translation model using T5x, we submit the following slurm job.
```
sbatch src/reference_implementations/t5x/run_multinode_2_2.slrm <path/to/a_t5x_workspace/dir> <path/to/a_t5x_log/dir>
```

Then, you can monitor the training status using `tensorboard` by specifying the directory used for saving models:
`MODEL_DIR` is defined at `src/reference_implementations/t5x/run_t5x.sh`

```
tensorboard --logdir=MODEL_DIR --bind_all
```

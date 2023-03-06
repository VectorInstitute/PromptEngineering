# Running T5x (Not Fully Supported During Prompt Engineering Lab)

We support running T5x only on the vector's cluster.
Make sure you install the cluster dependencies via the following command:
```
bash setup.sh OS=vcluster ENV_NAME=t5x DEV=false
```

To start training a translation model using T5x, we submit the following slurm job.
```
source t5x-env/bin/activate
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       <path/to/a_t5x_run_script.sh> \
       <path/to/a_t5x_log/dir> \
       <path/to/a_model_save/dir> \
       <path/to/a_data_save/dir>
```

For example:
```
source t5x-env/bin/activate
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       src/reference_implementations/t5x/run_t5x.sh \
       ./t5x-exps-logs \
       /scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/model \
       /scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/data
```

Then, you can monitor the training status using `tensorboard` by specifying the directory used for saving models:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/model
```

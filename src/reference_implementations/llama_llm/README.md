# Usage
## Building the environment
First run `setup_env.sh` to automatically create your environment.
`bash setup_env.sh`

## Launching the script with default prompts
Next, run the slurm script to queue your job on 4 A40s (default). Make sure to
pass in the correct path to the LLaMA checkpoint weights.
`sbatch launch_slurm.slrm /path/to/LLaMA`

You can check for the output of the model in the `/logs` directory that was
created for you.

## Launching the script with custom prompts
You can change the prompts used for inference by going into the
`llama/example.py` file located in the `git clone`'d LLaMA repository. In the
`main()` function, there is a list of strings called `prompts`. Feel free to
add or remove prompt strings. You can also extend this file to generate these
prompts in any way you choose.

## Multi-node use case
If you would like to run the largest version of LLaMA, then this section will
tell you how to do so. You'll first have to change `launch_slurm.slrm`:
`#SBATCH --nodes=2`
`MODEL_SIZE="65B"`

WARNING: The multi-node use case functions if you follow this section, but
LLaMA-65B is not rigoursly tested. Feel free to reach out to your technical
facilitator with any problems you may encounter.

LLaMA-65B is just more than double the size of LLaMA-30B, so we are taking
double the resources (GPUs) by asking for `--nodes=2`.

Next, we will need to patch some code within LLaMA since it's not configured
properly for our cluster. Open `llama/example.py` from the `git clone`'d LLaMA
repository, and go to the function called `setup_model_parallel()`. Delete the
function body and replace it with:
```
# Init mp
world_size = int(os.environ.get("SLURM_JOB_NUM_NODES", -1)) * int(os.environ.get("SLURM_GPUS_ON_NODE", -1))
torch.distributed.init_process_group("nccl")
initialize_model_parallel(world_size)

# Set local and global ranks
global_rank = torch.distributed.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", -1))
torch.cuda.set_device(local_rank)

# seed must be the same in all processes
torch.manual_seed(1)

return global_rank, world_size
```
Finally, run `sbatch launch_slurm.slrm /path/to/LLaMA` as we did before to run
the multi-node job.

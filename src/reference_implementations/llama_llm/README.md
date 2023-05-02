# [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) Usage

LLaMA is a smaller model than OPT-175B but has been trained for much longer. As suggested in the [Chinchilla Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf), many extremely large LMs may be under trained. LLaMA follows the suggested training scheme for "optimal" performance and is able to reproduce or exceed the performance level of the largest OPT/GPT models on many tasks, despite being much smaller.

## Building the environment

First run `setup_env.sh` to automatically create your environment. It is best to run this on a GPU node rather than the login nodes by first running
```
srun --gres=gpu:1 -c 8 --mem 16G -p t4v2 --pty bash
```
to secure a GPU then running `bash setup_env.sh`

## Launching the script with default prompts

Next, run the slurm script to queue your job on 4 A40s (default). Make sure to
pass in the correct path to the LLaMA checkpoint weights.

`sbatch launch_slurm.slrm /ssd003/projects/aieng/public/llama`

You can check for the output of the model in the `/logs` directory that was
created for you during your env install. __Make sure not to delete this folder, or else your jobs may fail.__

## Launching the script with custom prompts

You can change the prompts used for inference by going into the
`llama/example.py` file located in the `git clone`'d LLaMA repository. In the
`main()` function, there is a list of strings called `prompts`. Feel free to
add or remove prompt strings. You can also extend this file to generate these
prompts in any way you choose.

## Multi-node use case (Running LLaMA 65B)

If you would like to run the largest version of LLaMA, then this section will
tell you how to do so. You'll first have to change `launch_slurm.slrm`:

`#SBATCH --nodes=2`

`MODEL_SIZE="65B"`

<font color='red'> WARNING: The multi-node use case functions if you follow this section, but
LLaMA-65B is not rigoursly tested. Feel free to reach out to your technical
facilitator with any problems you may encounter.
</font>

LLaMA-65B is just more than double the size of LLaMA-30B, so we are taking
double the resources (GPUs) by asking for `--nodes=2`.

Next, we will need to patch some code within LLaMA since it's not configured
properly for our cluster. Open `llama/example.py` from the `git clone`'d LLaMA
repository, and go to the function called `setup_model_parallel()`. Delete the
function body and replace it with:
```python
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
Finally, run

`sbatch launch_slurm.slrm /ssd003/projects/aieng/public/llama`

as we did before to run the multi-node job.

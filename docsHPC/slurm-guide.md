# Cluster Guide

The Unibo guide on accessing the cluster can be found [here](https://disi.unibo.it/en/department/technical-and-administrative-services/it-services/cluster-hpc) (the page might have moved).

Folders:
* "home" folder (`~` or a.k.a. `/home/students/nome.cognome`), max 400MB, cache goes here by default so we have to use `--no-cache-dir` on all pip commands and similar things (e.g. see huggingface at the end of the file)
* `/scratch.hpc/nome.cognome`, unwritten agreement where you can create your own `nome.cognome` folder and work there with "unlimited" storage, just don't mess with other people's folders and projects

*Personal note*: I keep my project codes and slurm scripts in my home folder, while the virtualenvs in the `scratch.hpc` folder

## To move files from the cluster:
```
rsync -azvh --progress -e ssh nome.cognome@studio.unibo.it@giano.cs.unibo.it:filepath local_filepath_to_save_it
```

## To move files to the cluster:
```
rsync -azvh --progress -e ssh local_filepath nome.cognome@studio.unibo.it@giano.cs.unibo.it:filepath_to_save_it
```
**NOTE**: alternatively you can use github to pull from a repo that you push to regularly while developing your code / debugging / etc. - it's useful to use Personal Access Tokens to not have to login on github, by [creating a PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic) and then using `git clone https://<PAT>@github.com/username/repo.git` and simply `git pull` whenever needed.

## To create a virtual environment:

```shell
cd /scratch.hpc/nome.cognome/envs

python3 -m venv <nome_env> 

. /scratch.hpc/nome.cognome/envs/<nome_env>/bin/activate

pip3 install --no-cache-dir -r requirements.txt
pip3 install --no-cache-dir pandas
```
**NOTE**: (avoid the command `virtualenv <nome_env>`, see [here](https://www.reddit.com/r/learnpython/comments/4hsudz/pyvenv_vs_virtualenv/) for a discussion)

### Other useful linux commands :)
`du -sh .` shows memory occupation of the cwd, and `du -sh *` shows memory occupation of each file and directory in the cwd
`df -h` shows memory occupation of all memory partition
`nano filename` to modify a file from command line
`cat filename` or `cat > filename`, to show and create files
`head -20 filename`, to show the top 20 lines of a file


# Slurm Guide

## Starting a Job

The easiest way to lunch a slurm job is to define a `.sbatch` script that set all the job configurations first and then starts the actual job.

Here an example of python job, create the file `test.sbatch` with the following content:

```shell
#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nome.cognome@unibo.it   # or @studio.unibo.it
#SBATCH --time=07-00:00:00  # days-hours:minutes:seconds 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=l40
#SBATCH --gres=gpu:1

# create logs dir if it doesn't exist
mkdir -p logs

. /scratch.hpc/nome.cognome/<your_venv_path>/bin/activate

python3 -c "import torch; print(torch.cuda.is_available())"

# NOTE: tests should be run **in sequence** on the same job, as done below, instead of **in parallel** over different jobs otherwise you would be using all the cluster's gpus and filling the queue
# perform the hyperparameter search
for arg1 in A B C; do
    for arg2 in D E F; do
        python3 <your_home>/<project_path>/test.py --argument1 $arg1 --argument2 $arg2 > logs/slurm.$SLURM_JOB_ID.$arg1.$arg2.out 2> logs/slurm.$SLURM_JOB_ID.$arg1.$arg2.err
    done
done

```

The first part of the file instructs slurm about the job, defining the job nome, the type of notification we want to receive (if enabled), the number of nodes, and the tasks per node. We define the path where the stdout and stderr will be redirected (the `%N` and `%j` are slurm filename patterns to create distinc logs for each job, see more [here](https://slurm.schedmd.com/sbatch.html#SECTION_FILENAME-PATTERN))
In case we need a gpu we specify it using the paramter `--gres=gpu:1` that tells to add a gpu resource. Finally, the `--mem=0` allows using all the node's memory for the job.

For the full list of parameters check [here](https://slurm.schedmd.com/sbatch.html).

The second part of the file it's a regular shell script in which we start our job. In the example, we activate the python virtual environment and then we start a python script.

Finally, to start the job we type:

```shell
sbatch test.sbatch
```

## Slurm Utils

Here is a list of useful commands for managing and monitoring the slurm nodes and the running jobs.

### sbatch

The `sbatch <script_name>.sbatch` allows to start a sbatch script

### squeue

The `squeue` allows to see the running jobs. It is possible to filter out only the jobs of a specific user by adding the `-u <username>  argument.`

### scancel

The `scancel <job id>`  allows to terminate the running job with id `<job id>` (the id is given when the job is scheduled using  `sbatch <script_name>.sbatch` or it is visible using the `squeue` command).

### sinfo

The `sinfo` command shows information about the cluster nodes. If called without parameters it shows the number of nodes with their status. This command can be used also for showing all the resources available in the cluster: `sinfo -o "%20N  %10c  %10m  %25f  %10G "`.

## If you are using HuggingFace
At the beginning of your main.py, setup huggingface's cache in /scratch.hpc/nome.cognome by adding:
```python
import os
os.environ['HF_HOME'] = '/scratch.hpc/nome.cognome/cache/huggingface/'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/scratch.hpc/nome.cognome/cache/huggingface/'
```
Otherwise, it will fill your home folder immediately, since it has a max capacity of 400MB.
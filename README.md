# 2024 thesis
by Paul Lambert <lamber10 @ illinois . edu>
in collaboration with Prof Hockenmaier 

Pruning and quantization oh my

Current pipeline:

0. Launch an interactive session with GPUs:  
    * `srun --account=lamber10-ic --partition=secondary --gres=gpu:A40:2 --time=01:00:00 --pty /bin/bash`

1. Environment (can seed from `environment.yml`)
    * `conda activate res_env`
    * `resenv; resdir`

To run things with a batch script:  
`sbatch batch/init_model.sh llama321b boolq` 

A full list of valid shortnames can be found in `src/support/experiment_config.yaml`

To run things individually:  

1. Initialize a model   
    * `torchrun --nproc_per_node=1 src/1-gpu-collector.py --model llama321b --task boolq `
    * Use HuggingFace model name

2. Process the data  
    * `python src/2-meantensor.py --model llama321b --task boolq`

3. Look at the data (optional)
    * `python src/3-generate-viz.py --model=llama321b --task boolq`

4. Evaluate the model
   * ` python src/4-eval.py --model llama321b --task boolq --n 1` (where `n` is the number of layers to prune) 

Etc: 
    * `squeue -u $USER`
    * `watch -n 5 squeue -u $USER`

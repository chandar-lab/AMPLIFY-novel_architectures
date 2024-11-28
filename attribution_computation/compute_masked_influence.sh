#!/bin/bash
#SBATCH --job-name=masked_attribution_computation
#SBATCH --partition=lab-chandar
#SBATCH --output=logs/%j-%x.out
#SBATCH --error=logs/%j-%x.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64Gb

mkdir -p logs

if [[ -z "${AMPLIFY_PROJECT_DIRECTORY}" ]]; then
    echo "Environment variable `AMPLIFY_PROJECT_DIRECTORY` is not set! Exiting immediately."
    exit
fi

for masking_percentage in 0.1 0.15 0.2 0.25 0.3;
do
    srun ${AMPLIFY_PROJECT_DIRECTORY}//compute_masked_influence.py \
        --source mila \
        --model_name AMPLIFY350M \
        --model_path /network/projects/drugdiscovery/AMPLIFY/checkpoints/AMPLIFY_350M/pytorch_model.pt \
        # --tokenizer_path /network/projects/drugdiscovery/AMPLIFY/checkpoints/AMPLIFY_350M \
        --config_path ${AMPLIFY_PROJECT_DIRECTORY}/data/AMPLIFY_350_config.yaml
        --device cuda \
        --fp16 \
        # --n_proteins 10 \
        --dataset UniProt \
        --num_passes 100 \
        --mask_probability 0.15 \
        # --span_probability 0.15 --span_mask 5 \
        --exclude_special_tokens_replacement \
        --masked_only \
        --save_folder ${AMPLIFY_PROJECT_DIRECTORY}/attribution_computation/results \
        --seed 42
done
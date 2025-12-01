#!/bin/bash
#SBATCH --job-name=minigrid_uncertainty
#SBATCH --output=minigrid_uncertainty-%j.out
#SBATCH --error=minigrid_uncertainty-%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --gres=gpu:1


module purge
module load cuda11.3/toolkit/11.3.1

# set up micromamba environment
MAMBA_ROOT="/var/scratch/user/micromamba"
PATH="${MAMBA_ROOT}/bin:$PATH"
ENV_NAME="minari_env"
PYTHON="${MAMBA_ROOT}/envs/${ENV_NAME}/bin/python"

# set Minari datasets path
export MINARI_DATASETS_PATH="/var/scratch/user/.minari/datasets"

export CUDA_VISIBLE_DEVICES=0

# enable CUDA accelerated training
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# ultra-conservative training parameters to prevent collapse
LEARNING_RATE=3e-4   
BATCH_SIZE=16        
EPOCHS=30            
STEPS_PER_EPOCH=200  
EARLY_STOPPING=15    

# enhanced ensemble diversity parameters
ENSEMBLE_SIZE=5      
DROPOUT_RATE=0.2     
KL_WEIGHT=1e-4       

# network architecture
HIDDEN_DIMS="512,256,128"
N_QUANTILES=51       # good balance of quantiles

# enhanced methodology parameters
DIRECT_MEASUREMENT="--direct_measurement"
ADAPTIVE_METHOD="epistemic_weighted"
EPISTEMIC_WEIGHT=0.6
ALEATORIC_WEIGHT=0.4
INTERVAL_SCALE=0.3 
UNCERTAINTY_TEMP=2.0
VISUALIZATION_LEVEL="normal"

# dataset parameters
DATASET="minigrid/BabyAI-GoToLocal/optimal-v0"
DATA_PATH="/var/scratch/zwa212/.minari/datasets"
SAVE_DIR="outputs/minigrid_experiment_$(date +%Y%m%d_%H%M%S)/minigrid_experiment_$(date +%Y%m%d_%H%M%S)"

# uncertainty quantification parameters
COVERAGE_LEVEL=0.9
ALPHA=0.1

echo "Running with ultra-conservative parameters:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Ensemble Size: $ENSEMBLE_SIZE"
echo "  Dropout Rate: $DROPOUT_RATE"

# run the experiment with ultra-conservative parameters
${PYTHON} run_minigrid_experiment.py \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --save_dir "$SAVE_DIR" \
    --ensemble_size $ENSEMBLE_SIZE \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --steps_per_epoch $STEPS_PER_EPOCH \
    --early_stopping $EARLY_STOPPING \
    --hidden_dims "$HIDDEN_DIMS" \
    --n_quantiles $N_QUANTILES \
    --dropout_rate $DROPOUT_RATE \
    --kl_weight $KL_WEIGHT \
    --coverage_level $COVERAGE_LEVEL \
    --alpha $ALPHA \
    $DIRECT_MEASUREMENT \
    --adaptive_method "$ADAPTIVE_METHOD" \
    --epistemic_weight $EPISTEMIC_WEIGHT \
    --aleatoric_weight $ALEATORIC_WEIGHT \
    --interval_scale $INTERVAL_SCALE \
    --uncertainty_temp $UNCERTAINTY_TEMP \
    --visualization_level "$VISUALIZATION_LEVEL" \
    --device cuda \
    --seed 42

echo "=== Experiment completed. Check results in $SAVE_DIR ==="


if [ $? -ne 0 ]; then
    echo "Error in experiment. Check logs for details."
    exit 1
fi

echo "Experiment completed successfully!"
echo "Results saved to $SAVE_DIR"

# run the experiment with enhanced parameters
# python run_minigrid_experiment.py \
#     --dataset "minigrid/BabyAI-GoToLocal/optimal-v0" \
#     --data_path "/var/scratch/zwa212/.minari/datasets" \
#     --save_dir "outputs/minigrid_experiment_$(date +%Y%m%d_%H%M%S)" \
#     --epochs 50 \
#     --batch_size 16 \
#     --steps_per_epoch 300 \
#     --learning_rate 2e-5 \
#     --ensemble_size 3 \
#     --n_quantiles 51 \
#     --hidden_dims "256,128,128" \
#     --dropout_rate 0.1 \
#     --kl_weight 1e-5 \
#     --early_stopping 15 \
#     --epistemic_weight 0.6 \
#     --aleatoric_weight 0.4 \
#     --alpha 0.1 \
#     --coverage_level 0.9 \
#     --interval_scale 0.3 \
#     --adaptive_method "epistemic_weighted" \
#     --visualization_level "detailed" \
#     --uncertainty_temp 1.0 \
#     --seed 42

echo "=== Experiment completed ==="
echo "Check the output directory for results and visualizations" 
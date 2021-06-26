{
    set -e
    source /home/liz0f/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate pytorch1.6
    CUDA_VISIBLE_DEVICES=1 python main.py eval --config configs/ICDAR.yaml
}
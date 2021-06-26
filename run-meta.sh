{
    set -e
    source /home/liz0f/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate pytorch1.6

    exp_root_dir="./ICDAR-hyper-search"
    python prepare_meta_experiment.py \
    --tune-hyper-parameters lr,batch_size,loss_fn \
    --meta-config configs/ICDAR-meta.yaml \
    --exp-root-dir "$exp_root_dir" \

    for config_fp in "${exp_root_dir}"/configs/*.yaml; do
        bn="$(basename $config_fp)"
        exp_code="${bn%.*}"
        echo "$exp_code"
        CUDA_VISIBLE_DEVICES=0 python main.py train --config "$config_fp"
    done

    exit 0;
}
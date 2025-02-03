MODEL_PATH=~/hydra/singlerun/2024-11-07/mp_20_1e3_0.5_s1000_sim_dtime_cate_ema0.995cs_11-07-22-28-31
export CUDA_VISIBLE_DEVICES=1

LABEL=end_back
python scripts/evaluate_vmbfn.py --num_batches_to_samples 10 --batch_size 1000\
            --label $LABEL\
            --model_path $MODEL_PATH --tasks gen --samp_acc_factor 1 --n_step_each 1000\

python scripts/compute_metrics.py --root_path $MODEL_PATH --tasks gen --stability --label $LABEL

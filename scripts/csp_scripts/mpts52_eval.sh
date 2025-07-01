MODEL_PATH=/data/wuhl/CrysBFN/hydra/singlerun/mpts_csp_1e3_0.5_s500_sim_ema0.995_09-27-10-26-37
strategy='end_back'

LABEL='default'
export CUDA_VISIBLE_DEVICES=0

python scripts/evaluate_crysbfn.py --batch_size 1000\
            --label $LABEL --strategy $strategy\
            --model_path $MODEL_PATH --tasks csp --n_step_each 1000 --num_batches_to_csp 100
python scripts/compute_metrics.py --root_path $MODEL_PATH --tasks csp --label $LABEL

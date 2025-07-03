MODEL_PATH=/data/wuhl/CrysBFN/hydra/mpts_csp_s500
strategy='end_back'

LABEL='default'
export CUDA_VISIBLE_DEVICES=0

python scripts/evaluate_crysbfn.py --batch_size 1000\
            --label $LABEL --strategy $strategy\
            --model_path $MODEL_PATH --tasks csp --num_batches_to_csp 100 # we set a maximum of 100 batches to csp while 
python scripts/compute_metrics.py --root_path $MODEL_PATH --tasks csp --label $LABEL

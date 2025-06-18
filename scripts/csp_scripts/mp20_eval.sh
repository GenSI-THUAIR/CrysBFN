MODEL_PATH=YOUR_MODEL_PATH
strategy='end_back'

LABEL=${strategy}
export CUDA_VISIBLE_DEVICES=0

python scripts/evaluate_vmbfn.py --num_batches_to_csp 10 --batch_size 1000\
            --label $LABEL --strategy $strategy\
            --model_path $MODEL_PATH --tasks csp --samp_acc_factor 1 --n_step_each 1000 
python scripts/compute_metrics.py --root_path $MODEL_PATH --tasks csp --label $LABEL


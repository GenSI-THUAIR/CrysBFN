MODEL_PATH=/data/wuhl/crysbfn_reimplement/CrysBFN/hydra/mp_20_gen_s1000
export CUDA_VISIBLE_DEVICES=1

LABEL=end_back
python scripts/evaluate_crysbfn.py --num_batches_to_samples 10 --batch_size 1000\
            --label $LABEL\
            --model_path $MODEL_PATH --tasks gen\

python scripts/compute_metrics.py --root_path $MODEL_PATH --tasks gen --label $LABEL

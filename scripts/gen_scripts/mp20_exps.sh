STEPS=1000
BETA1_COORD=1e3
BETA1_TYPE=0.5
SIGMA1_LATTICE=0.03162277660168379
SIM_FLOW=true
RESUME=false

export RAY_memory_usage_threshold=0.999

disc_prob_loss=true
if [ "$disc_prob_loss" = true ]; then
    disc_prob=dtime_cate
else
    disc_prob=ctime_cate
fi

USE_EMA=true
EMA_DECAY=0.995
if [ "$USE_EMA" = true ]; then
    EMA_STR=ema${EMA_DECAY}cs
else
    EMA_STR=noema
fi

DATASET=mp_20

GPU_ID_LIST=(5)
BETA1_TYPES=(0.5 0.4 1.0)
EXP_STEPS=(1000)

for ((index = 0; index < ${#GPU_ID_LIST[@]}; index++))
do
    STEPS=${EXP_STEPS[index]}
    current_time=$(date +"%m-%d-%H-%M-%S")
    EXPNAME=${DATASET}_${BETA1_COORD}_${BETA1_TYPE}_s${STEPS}_sim_${disc_prob}_${EMA_STR}
    EXPNAME=${EXPNAME}_${current_time}
    echo "EXPNAME: $EXPNAME"
    export CUDA_VISIBLE_DEVICES=${GPU_ID_LIST[index]}
    nohup python crysbfn/run.py \
        data=${DATASET} expname=$EXPNAME train.resume=$RESUME \
        model.BFN.dtime_loss_steps=$STEPS\
        model.BFN.beta1_coord=$BETA1_COORD model.BFN.beta1_type=${BETA1_TYPE}\
        logging.gen_check.start_epoch=800 logging.gen_check_interval=120 logging.gen_check.num_samples=128\
        train.ema.enable=$USE_EMA train.ema.decay=$EMA_DECAY\
        model.BFN.disc_prob_loss=$disc_prob_loss\
        model.BFN.sigma1_lattice=$SIGMA1_LATTICE optim.lr_scheduler.patience=50 > ./logs/${EXPNAME}.log&\
done
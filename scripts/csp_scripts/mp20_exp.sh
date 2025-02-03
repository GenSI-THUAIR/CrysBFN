STEPS=1000
BETA1_COORD=1e3
BETA1_TYPE=0.5
SIGMA1_LATTICE=0.03162277660168379
RESUME=false
export RAY_memory_usage_threshold=0.999

USE_EMA=true
EMA_DECAY=0.995
EMA_STR=ema${EMA_DECAY}

SIM_FLOW=true
DATASET=mp_20

EXP_STEPS=(50)
GPU_ID_LIST=(0)

for ((index = 0; index < ${#GPU_ID_LIST[@]}; index++))
do
    STEPS=${EXP_STEPS[index]}
    current_time=$(date +"%m-%d-%H-%M-%S")
    EXPNAME=${DATASET}_csp_${BETA1_COORD}_${BETA1_TYPE}_s${STEPS}_$SIM_STR_${EMA_STR}
    EXPNAME=${EXPNAME}_${current_time}
    echo "EXPNAME: $EXPNAME"
    export CUDA_VISIBLE_DEVICES=${GPU_ID_LIST[index]}
    nohup python crysbfn/run.py \
        model=bfn_csp\
        logging.wandb.project=crystalbfn_csp\
        data=${DATASET} expname=$EXPNAME train.resume=$RESUME \
        model.BFN.dtime_loss_steps=$STEPS\
        model.BFN.beta1_coord=$BETA1_COORD model.BFN.beta1_type=${BETA1_TYPE}\
        logging.gen_check.start_epoch=300 logging.gen_check_interval=100\
        train.ema.enable=$USE_EMA train.ema.decay=$EMA_DECAY\
        model.BFN.disc_prob_loss=$disc_prob_loss\
        model.BFN.sim_cir_flow=$SIM_FLOW\
        model.BFN.sigma1_lattice=$SIGMA1_LATTICE optim.lr_scheduler.patience=50 > ./logs/${EXPNAME}.log&\
done
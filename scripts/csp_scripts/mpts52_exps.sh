BETA1_COORD=1e3
BETA1_TYPE=0.5
SIGMA1_LATTICE=0.03162277660168379

RESUME=false

USE_EMA=true
EMA_DECAY=0.995
if [ "$USE_EMA" = true ]; then
    EMA_STR=ema${EMA_DECAY}
else
    EMA_STR=noema
fi

DATASET=mpts

SIM_FLOW=true
if [ "$SIM_FLOW" = true ]; then
    SIM_STR=sim
else
    SIM_STR=nosim
fi
GPU_ID_LIST=(0)
EXP_STEPS=(200)

export WANDB_BASE_URL=https://api.bandw.top # if you can use wandb directly, you can comment this line
for ((index = 0; index < ${#GPU_ID_LIST[@]}; index++))
do
    STEPS=${EXP_STEPS[index]}
    current_time=$(date +"%m-%d-%H-%M-%S")
    EXPNAME=${DATASET}_csp_${BETA1_COORD}_${BETA1_TYPE}_s${STEPS}_${SIM_STR}_${EMA_STR}
    EXPNAME=${EXPNAME}_${current_time}
    echo "EXPNAME: $EXPNAME"
    export CUDA_VISIBLE_DEVICES=${GPU_ID_LIST[index]}
    nohup python crysbfn/run.py \
        model=bfn_csp\
        logging.wandb.project=crystalbfn_csp\
        data=${DATASET} expname=$EXPNAME train.resume=$RESUME \
        model.BFN.dtime_loss_steps=$STEPS\
        model.BFN.beta1_coord=$BETA1_COORD model.BFN.beta1_type=${BETA1_TYPE}\
        logging.gen_check.start_epoch=400 logging.gen_check_interval=30 logging.gen_check.num_samples=10000\
        train.ema.enable=$USE_EMA train.ema.decay=$EMA_DECAY\
        model.BFN.sim_cir_flow=$SIM_FLOW\
        model.BFN.disc_prob_loss=$disc_prob_loss\
        model.BFN.sigma1_lattice=$SIGMA1_LATTICE optim.lr_scheduler.patience=50 > ./logs/${EXPNAME}.log&\
done
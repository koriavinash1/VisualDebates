python ../main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/MNIST \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/MNIST/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/MNIST/debate/plots \
        --img_size=32 \
        --model=vanilla \
        --epoch=35 \
        --batch_size=64 \
        --device=0 \
        --print_freq=500 \
        --plot_freq=50 \
        --n_class=10 \
        --init_lr=1e-3 \
        --patch_size=$1 \
        --nagents=$2 \
        --narguments=$3 \
        --nglimpses=$4 \
        --rnn_hidden=$5 \
        --glimpse_hidden=$6 \
        --loc_hidden=$7 \
        --reward_weightage=$8
                    
echo Support Pre-training Completed

python ../main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/MNIST \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/MNIST/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/MNIST/debate/plots \
        --img_size=32 \
        --model=vanilla \
        --epoch=50 \
        --batch_size=64 \
        --device=0 \
        --print_freq=100 \
        --plot_freq=50 \
        --n_class=10 \
        --init_lr=1e-5 \
        --contrastive=True \
        --resume=True \
        --patch_size=$1 \
        --nagents=$2 \
        --narguments=$3 \
        --nglimpses=$4 \
        --rnn_hidden=$5 \
        --glimpse_hidden=$6 \
        --loc_hidden=$7 \
        --reward_weightage=$8
                    
echo Debate Completed
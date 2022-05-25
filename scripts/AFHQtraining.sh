python ../main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/AFHQ/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/AFHQ/debate/plots \
        --img_size=128 \
        --model=densenet121 \
        --epoch=250 \
        --batch_size=16 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=50 \
        --n_class=3 \
        --init_lr=5e-3 \
        --patch_size=$1 \
        --nagents=$2 \
        --narguments=$3 \
        --nglimpses=$4 \
        --rnn_hidden=$5 \
        --glimpse_hidden=$6 \
        --loc_hidden=$7 \
        --reward_weightage=$8

echo Debate Completed

python ../main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/AFHQ/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/AFHQ/debate/plots \
        --img_size=128 \
        --model=densenet121 \
        --epoch=300 \
        --batch_size=16 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=50 \
        --n_class=3 \
        --init_lr=5e-4 \
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
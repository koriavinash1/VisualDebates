python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsS/AFHQ/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsS/AFHQ/debate/plots \
        --img_size=128 \
        --epoch=150 \
        --batch_size=16 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=3 \
        --n_class=3 \
        --nfeatures=1024\
        --cdim=16 \
        --init_lr=1e-3 \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4

echo Debate Completed

python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsS/AFHQ/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsS/AFHQ/debate/plots \
        --img_size=128 \
        --epoch=200 \
        --batch_size=16 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=3 \
        --n_class=3 \
        --init_lr=1e-5 \
        --contrastive=True \
        --resume=True \
        --nfeatures=1024 \
        --cdim=16 \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4

echo Debate Completed
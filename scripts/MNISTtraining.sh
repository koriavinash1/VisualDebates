python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/MNIST \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsS2/MNIST/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsS2/MNIST/debate/plots \
        --img_size=32 \
        --epoch=50 \
        --batch_size=64 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --n_class=10 \
        --init_lr=1e-3 \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4
                    
echo Support Pre-training Completed

python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/MNIST \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsS2/MNIST/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsS2/MNIST/debate/plots \
        --img_size=32 \
        --epoch=75 \
        --batch_size=64 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --n_class=10 \
        --init_lr=1e-5 \
        --contrastive=True \
        --resume=True \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4
                    
echo Debate Completed
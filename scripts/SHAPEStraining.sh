python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/SHAPES/shapes \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/SHAPES/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/SHAPES/debate/plots \
        --img_size=32 \
        --epoch=25 \
        --batch_size=64 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --n_class=4 \
        --init_lr=5e-4 \
        --nfeatures=32 \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4
                    
echo Support Pre-training Completed

python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/SHAPES/shapes \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/SHAPES/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/SHAPES/debate/plots \
        --img_size=32 \
        --epoch=40 \
        --batch_size=64 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --n_class=4 \
        --init_lr=2e-5 \
        --contrastive=True \
        --resume=True \
        --nfeatures=32 \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4
                    
echo Debate Completed
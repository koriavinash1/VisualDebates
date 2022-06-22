python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/MNIST \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/MNIST/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/MNIST/debate/plots \
        --img_size=32 \
        --epoch=25 \
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
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/MNIST/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/MNIST/debate/plots \
        --img_size=32 \
        --epoch=40 \
        --batch_size=64 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --n_class=10 \
        --init_lr=1e-4 \
        --contrastive=True \
        --resume=True \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4
                    
echo Debate Completed


python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/src/plots.py \
       --plot_dir /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/MNIST/debate/plots/Exp-test-Debate:GRU_$1_$4_$2_$3_1/Exp-test-Debate:GRU_$1_$4_$2_$3_1 \
       --start_epoch 0 \
       --start_epoch 39 \
       --split_epoch 24
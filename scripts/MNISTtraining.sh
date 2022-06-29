NAME='test'
LOGSDIR='/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsConstrastive/MNIST/'$NAME


python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/MNIST \
        --name=$NAME \
        --ckpt_dir=$LOGSDIR/ckpt \
        --plot_dir=$LOGSDIR/plots \
        --img_size=32 \
        --epoch=25 \
        --batch_size=32 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --n_class=10 \
        --init_lr=1e-3 \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4 \
        --quantize=spatial \
        # --include_classes '9, 6' \
                    
echo Support Pre-training Completed

python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/MNIST \
        --name=$NAME \
        --ckpt_dir=$LOGSDIR/ckpt \
        --plot_dir=$LOGSDIR/plots \
        --img_size=32 \
        --epoch=40 \
        --batch_size=32 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --n_class=10 \
        --init_lr=1e-3 \
        --contrastive=True \
        --resume=True \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4 \
        --quantize=spatial \
        # --include_classes '9, 6' \

                    
echo Debate Completed


echo Plotting logs from:---------
echo $LOGSDIR/plots/Exp-$NAME-Debate:GRU_$1_$4_$2_$3_1/Exp-$NAME-Debate:GRU_$1_$4_$2_$3_1

python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/src/plots.py \
       --plot_dir $LOGSDIR/plots/Exp-$NAME-Debate:GRU_$1_$4_$2_$3_1/Exp-$NAME-Debate:GRU_$1_$4_$2_$3_1 \
       --quantize=spatial \
       --start_epoch 0 \
       --stop_epoch 39 \
       --split_epoch 24
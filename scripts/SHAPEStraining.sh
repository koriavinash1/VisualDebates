NAME='test'
QUANTIZE='spatial'
LOGSDIR='/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsContrastive2/SHAPES/'$NAME

python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/SHAPES/shapes \
        --name=$NAME \
        --ckpt_dir=$LOGSDIR/ckpt \
        --plot_dir=$LOGSDIR/plots \
        --img_size=32 \
        --epoch=35 \
        --batch_size=32 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --init_lr=1e-3 \
        --nfeatures=32 \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4 \
        --n_class=4 \
        --modulated_channels=32 \
        --quantize=$QUANTIZE
        # --include_classes 'circle, star' 
                    
echo Support Pre-training Completed

python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/SHAPES/shapes \
        --name=$NAME \
        --ckpt_dir=$LOGSDIR/ckpt \
        --plot_dir=$LOGSDIR/plots \
        --img_size=32 \
        --epoch=50 \
        --batch_size=32 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --contrastive=True \
        --init_lr=1e-3 \
        --resume=True \
        --nfeatures=32 \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4 \
        --n_class=4 \
        --modulated_channels=32 \
        --quantize=$QUANTIZE 
        # --include_classes 'circle, star' 
                    
echo Debate Completed


echo Plotting logs from:---------
echo $LOGSDIR/plots/Exp-$NAME-Debate:GRU_$1_$4_$2_$3_1/Exp-$NAME-Debate:GRU_$1_$4_$2_$3_1

python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/src/plots.py \
       --plot_dir $LOGSDIR/plots/Exp-$NAME-Debate:GRU_$1_$4_$2_$3_1/Exp-$NAME-Debate:GRU_$1_$4_$2_$3_1 \
       --quantize=$QUANTIZE \
       --start_epoch 0 \
       --stop_epoch 49 \
       --split_epoch 34
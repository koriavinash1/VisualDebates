# python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
#         --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
#         --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS1/AFHQ/debate/ckpt \
#         --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS1/AFHQ/debate/plots \
#         --img_size=128 \
#         --epoch=30 \
#         --batch_size=16 \
#         --device=0 \
#         --print_freq=50 \
#         --plot_freq=1 \
#         --n_class=3 \
#         --nfeatures=1024\
#         --cdim=16 \
#         --init_lr=1e-3 \
#         --disentangle=True \
#         --narguments=$1 \
#         --rnn_hidden=$2 \
#         --rnn_input_size=$3 \
#         --nconcepts=$4

# echo Debate Completed

python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS1/AFHQ/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS1/AFHQ/debate/plots \
        --img_size=128 \
        --epoch=50 \
        --batch_size=16 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=1 \
        --n_class=3 \
        --init_lr=1e-5 \
        --contrastive=True \
        --resume=True \
        --nfeatures=1024 \
        --cdim=16 \
        --disentangle=True \
        --narguments=$1 \
        --rnn_hidden=$2 \
        --rnn_input_size=$3 \
        --nconcepts=$4

echo Debate Completed


python /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/src/plots.py \
       --plot_dir /vol/biomedic2/agk21/PhDLogs/codes/AIDebatesOnSymbols/LogsSS2/AFHQ/debate/plots/Exp-test-Debate:GRU_$1_$4_$2_$3_1/Exp-test-Debate:GRU_$1_$4_$2_$3_1 \
       --start_epoch 0 \
       --start_epoch 49 \
       --split_epoch 29
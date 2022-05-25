python ../main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/MNIST \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/MNIST/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/MNIST/debate/plots \
        --batch_size=32 \
        --model=vanilla \
        --img_size=32 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=50 \
        --n_class=10 \
        --reward_weightage=1 \
        --nagents=2 \
        --nglimpses=1 \
        --patch_size=4 \
        --narguments=6 \
        --rnn_hidden=128 \
        --loc_hidden=64 \
        --num_plots=32 \
        --is_train False \
        --glimpse_hidden=64 \
        --contrastive=True \
        --is_plot True \
        --best True \

python3 ../src/plot_glimpses.py \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/MNIST/debate/plots/vanilla_RNN_6_4x4_2_1_64_64_128_1.0 \
        --epoch=-1 \
        --patch_size=4 \
        --nagents=2 \
        --nglimpses=1 \
        --name=plots
echo Done plots 1


python ../main.py \
        --data_dir=/vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
        --ckpt_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/AFHQ/debate/ckpt \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/AFHQ/debate/plots \
        --img_size=32 \
        --model=densenet121 \
        --batch_size=32 \
        --device=0 \
        --print_freq=50 \
        --plot_freq=50 \
        --n_class=3 \
        --nagents=2 \
        --narguments=6 \
        --patch_size=16 \
        --nglimpses=1 \
        --num_plots=32 \
        --rnn_hidden=256 \
        --glimpse_hidden=128 \
        --loc_hidden=128 \
        --reward_weightage=1 \
        --is_train False \
        --contrastive=True \
        --is_plot True \
        --best True \

python3 ../src/plot_glimpses.py \
        --plot_dir=/vol/biomedic2/agk21/PhDLogs/codes/Agent-debates/LogsS/AFHQ/debate/plots/densenet121_RNN_6_16x16_2_1_128_128_256_1.0 \
        --epoch=-1 \
        --patch_size=16 \
        --nagents=2 \
        --nglimpses=1 \
        --name=plots
echo Done plots 2
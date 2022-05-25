# python train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
#         --nclasses 3 \
#         --batch_size 16 \
#         --input_size 128 \
#         --model densenet121 \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/stylEX-extention/Classifier/Logs \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 250 \

        
python train.py \
        --data_root /vol/biomedic2/agk21/PhDLogs/datasets/MorphoMNIST/data  \
        --nclasses 10 \
        --batch_size 32 \
        --input_size 32 \
        --model vanilla \
        --seed 2022 \
        --logdir /vol/biomedic2/agk21/PhDLogs/codes/stylEX-extention/Classifier/Logs/MNIST \
        --decreasing_lr '20, 30, 40' \
        --epochs 50 \
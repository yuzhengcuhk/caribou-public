epoch=100
repeats=3
dropout=0.5

# modify chains.py to use "chains-1"

for dataset in pubmed
do
    for hops in 20 17 15 13 11 10 9 8 7 6 5 4 3 2 1 
    do
        echo "the value of hops is $hops ; the dataset is $dataset"
        python train.py gap-inf \
        --dataset $dataset \
        --encoder_layers 2 \
        --base_layers 1 \
        --head_layers 1 \
        --combine cat \
        --hops $hops \
        --hidden_dim 64 \
        --activation selu \
        --optimizer adam \
        --learning_rate 1e-3 \
        --repeats $repeats \
        --batch_norm True \
        --epochs $epoch \
        --batch_size full \
        --dropout $dropout \
        --encoder_epochs $epoch 
            #--device $device \
            #--project GAP/edge/$epsilon/$hops 
    done
done




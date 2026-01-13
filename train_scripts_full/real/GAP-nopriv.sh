device=$1
epoch=100
hops=2


for dataset in facebook cora pubmed photo computers squirrel chameleon
do
    for hops in 1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 50
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
        --learning_rate 0.001 \
        --repeats 1 \
        --batch_norm True \
        --epochs $epoch \
        --batch_size 64 \
        --dropout 0.5 \
        --encoder_epochs $epoch \
        --device cuda \
        --project GAP/nopriv/$hops 
    done
done



for dataset in squirrel #facebook cora pubmed photo computers squirrel chameleon
do
    OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=$device python train.py gap-inf \
    --dataset $dataset \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 0.001 \
    --repeats 10 \
    --batch_norm True \
    --epochs $epoch \
    --batch_size 64 \
    --dropout 0.5 \
    --encoder_epochs $epoch \
    --device cuda \
    --project GAP/nopriv/$hops
done


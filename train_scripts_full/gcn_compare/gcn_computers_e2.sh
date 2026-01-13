epoch=100
repeats=3
dropout=0.5
activation=selu
device=cuda:0
dataset=computers
epsilon=2

for hops in 15 10 7 4 2
do
    for bound_lipschitz in 0.6 0.7 0.8 0.9
    do
        for fill_degree in 5 10 50 100 500 1000 3000 5000 10000 20000
        do
            echo "Hello! Hello! Hello! (NO-MLP) *********** the value of epsilon is $epsilon ; the value of hops is $hops ; the dataset is $dataset ; the value of bound_lipschitz is $bound_lipschitz ; the value of fill_degree is $fill_degree"
            python train.py gcn-edp \
            --dataset $dataset \
            --epsilon $epsilon \
            --encoder_layers 2 \
            --base_layers 1 \
            --head_layers 1 \
            --combine cat \
            --hops $hops \
            --hidden_dim 64 \
            --activation $activation \
            --optimizer adam \
            --learning_rate 1e-3 \
            --repeats $repeats \
            --batch_norm True \
            --epochs $epoch \
            --batch_size full \
            --dropout $dropout \
            --encoder_epochs $epoch \
            --bound_lipschitz $bound_lipschitz\
            --fill_degree $fill_degree\
            --device $device\
            # --project GAP/edge/$epsilon/$hops
        done
    done
done





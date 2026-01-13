epoch=100
repeats=3
dropout=0.5
activation=relu
device=cuda:1
dataset=facebook


for epsilon in 1 2 4 8 16 32
do
    for hops in 100 70 50 45 40 35 30 25 20 15 10 9 8 7 6 5 4 3 2 1
    do
        for bound_lipschitz in 0.6 0.7 0.8 0.9
        do
            for fill_degree in 1 3 5 7 10 15 20 30 50 75 100
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
done





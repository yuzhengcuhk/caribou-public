epoch=100
repeats=3
dropout=0.5
epsilon=1

# modify chains.py to use "chains-1"

for dataset in facebook
do
    for hops in 20 17 15 13 11 10 9 8 7 6 5 4 3 2 1 
    do
        for bound_lipschitz in 0.9 0.8 0.7 0.6
        do
            for beta in 0.1 1 10
            do
                for alpha_1 in 1 0.9 
                do  
                    echo "the value of epsilon is $epsilon ; the value of hops is $hops ; the dataset is $dataset"
                    echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
                    python train.py gcn-edp \
                    --dataset $dataset \
                    --epsilon $epsilon \
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
                    --encoder_epochs $epoch \
                    --alpha_1 $alpha_1 \
                    --beta $beta \
                    --bound_lipschitz $bound_lipschitz \
                    --device cuda:1
            #--device $device \
            #--project GAP/edge/$epsilon/$hops 
                done
            done    
        done
    done
done




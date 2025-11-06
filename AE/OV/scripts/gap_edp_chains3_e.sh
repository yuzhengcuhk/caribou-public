epoch=100
    repeats=3
    dropout=0.5
    epsilon=1

    # Generated from CSV: epsilon=1.0, hops=13

    for dataset in chains3
    do
        hops=13
        
        echo "the value of epsilon is 1 ; the value of hops is $hops ; the dataset is $dataset"
        echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
        python train.py gap-edp \
        --dataset $dataset \
        --epsilon 1 \
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
        --dropout $dropout \
        --encoder_epochs $epoch \
        --device cuda
    done


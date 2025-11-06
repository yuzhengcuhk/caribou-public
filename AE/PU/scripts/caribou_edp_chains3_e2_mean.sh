epoch=100
    repeats=3
    dropout=0.5
    epsilon=2

    # Generated from CSV: epsilon=2.0, hops=20, beta=10.0, alpha_1=1.0, bound_lipschitz=0.6

    for dataset in chains3
    do
        hops=20
        beta=10.0
        alpha_1=1.0
        bound_lipschitz=0.6
        
        echo "the value of epsilon is 2 ; the value of hops is $hops ; the dataset is $dataset"
        echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
        python train.py caribou-edp \
        --dataset $dataset \
        --epsilon 2 \
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
        --device cuda
    done
    
    
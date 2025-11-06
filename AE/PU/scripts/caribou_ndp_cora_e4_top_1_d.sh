epoch=100
    repeats=3
    dropout=0.5
    epsilon=4

    # Generated from CSV: epsilon=4.0, hops=3, beta=20.0, alpha_1=1.0, bound_lipschitz=0.7

    for dataset in cora
    do
        hops=3
        beta=20.0
        alpha_1=1.0
        bound_lipschitz=0.7
        
        echo "the value of epsilon is 4 ; the value of hops is $hops ; the dataset is $dataset"
        echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
        python train.py caribou-ndp \
        --dataset $dataset \
        --epsilon 4 \
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
        --device cuda \
        --max_degree 10
    done
    
    
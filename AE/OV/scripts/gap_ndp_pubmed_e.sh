epoch=100
    repeats=3
    dropout=0.5
    epsilon=1

    # Generated from CSV: epsilon=1.0, hops=2

    for dataset in pubmed
    do
        hops=2
        
        echo "the value of epsilon is 1 ; the value of hops is $hops ; the dataset is $dataset"
        echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
        python train.py gap-ndp \
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
        --epochs $epoch \
        --dropout $dropout \
        --encoder_epochs $epoch \
        --device cuda \
        --max_degree 20
    done

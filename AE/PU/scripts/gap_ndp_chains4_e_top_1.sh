epoch=100
    repeats=3
    dropout=0.5
    epsilon=1

    # Generated from CSV: epsilon=1.0, hops=9

    for dataset in chains4
    do
        hops=9
        
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

    epoch=100
    repeats=3
    dropout=0.5
    epsilon=2

    # Generated from CSV: epsilon=2.0, hops=9

    for dataset in chains4
    do
        hops=9
        
        echo "the value of epsilon is 2 ; the value of hops is $hops ; the dataset is $dataset"
        echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
        python train.py gap-ndp \
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
        --epochs $epoch \
        --dropout $dropout \
        --encoder_epochs $epoch \
        --device cuda \
        --max_degree 20
    done

    epoch=100
    repeats=3
    dropout=0.5
    epsilon=4

    # Generated from CSV: epsilon=4.0, hops=9

    for dataset in chains4
    do
        hops=9
        
        echo "the value of epsilon is 4 ; the value of hops is $hops ; the dataset is $dataset"
        echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
        python train.py gap-ndp \
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
        --epochs $epoch \
        --dropout $dropout \
        --encoder_epochs $epoch \
        --device cuda \
        --max_degree 20
    done

    epoch=100
    repeats=3
    dropout=0.5
    epsilon=8

    # Generated from CSV: epsilon=8.0, hops=9

    for dataset in chains4
    do
        hops=9
        
        echo "the value of epsilon is 8 ; the value of hops is $hops ; the dataset is $dataset"
        echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
        python train.py gap-ndp \
        --dataset $dataset \
        --epsilon 8 \
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

    epoch=100
    repeats=3
    dropout=0.5
    epsilon=16

    # Generated from CSV: epsilon=16.0, hops=9

    for dataset in chains4
    do
        hops=9
        
        echo "the value of epsilon is 16 ; the value of hops is $hops ; the dataset is $dataset"
        echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
        python train.py gap-ndp \
        --dataset $dataset \
        --epsilon 16 \
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

    epoch=100
    repeats=3
    dropout=0.5
    epsilon=32

    # Generated from CSV: epsilon=32.0, hops=9

    for dataset in chains4
    do
        hops=9
        
        echo "the value of epsilon is 32 ; the value of hops is $hops ; the dataset is $dataset"
        echo "the value of beta is $beta ; the value of alpha_1 is $alpha_1 ; the bound_lipschitz is $bound_lipschitz"
        python train.py gap-ndp \
        --dataset $dataset \
        --epsilon 32 \
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

    
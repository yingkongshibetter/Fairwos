
echo '============bail_gin============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gin --dataset bail --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 5 --weight 5 --unit 6 --epochs_fair 15

echo '============bail_gcn============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset bail --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 5 --weight 5 --unit 6 --epochs_fair 15

echo '============pokec_n_gcn============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset pokec_n --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 5 --weight 5 --unit 16 --epochs_fair 1

echo '============pokec_n_gin============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gin --dataset pokec_n --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 5 --weight 1 --unit  16 --epochs_fair 1
echo '============pokec_z_gcn============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset pokec_z --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 10 --weight  3 --unit 2 --epochs_fair 3

echo '============pokec_z_gin============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gin --dataset pokec_z --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 10 --weight  10 --unit 4 --epochs_fair 15



echo '============occupation_gcn============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset occupation --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 1 --weight 1 --unit 32 --epochs_fair 25


echo '============occupation_gin============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gin --dataset occupation --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 5 --weight 5 --unit 8 --epochs_fair 15


echo '============NBA_gcn============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset nba --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 2 --weight  1 --unit 8 --epochs_fair 15
echo '============NBA_gin============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gin --dataset nba --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 2 --weight  0.25 --unit 8 --epochs_fair 15
echo '============credit_gcn============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset credit --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 5 --weight 5 --unit 3 --epochs_fair 15
echo '============credit_gin============='
CUDA_VISIBLE_DEVICES=0 python trainGNN_bail.py ---dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gin --dataset credit --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1 --indice 5 --weight 5 --unit 8 --epochs_fair 15
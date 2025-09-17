#python train.py --config configs/icod_train.py --pretrained --model-name CFFKDNet_RES2NET50
#python train.py --config configs/icod_train.py --pretrained --model-name MCDNet_PVTB2
#python train.py --config configs/icod_train.py --pretrained --model-name MCDNet_PVTB4
#python train.py --config configs/icod_train.py --pretrained --model-name MCDNet_EFFB4

python train_kd.py --config configs/icod_train.py --pretrained --model-name Res2Net50_KD
#python train_kd.py --config configs/icod_train.py --pretrained --model-name PVTB2_KD
#python train_kd.py --config configs/icod_train.py --pretrained --model-name EffB4_KD








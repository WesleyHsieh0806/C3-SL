# Run all the experiments (different gamma)
# python train_init.py --batch 64 --epoch 10
python train.py --batch 64 --epoch 15 --gamma 0.1
python train.py --batch 64 --epoch 15 --gamma 0.2
python train.py --batch 64 --epoch 15 --gamma 0.3
python train.py --batch 64 --epoch 15 --gamma 0.4
python train.py --batch 64 --epoch 15 --gamma 0.5
python train.py --batch 64 --epoch 15 --gamma 0.6
python train.py --batch 64 --epoch 15 --gamma 0.7
python train.py --batch 64 --epoch 15 --gamma 0.8
python train.py --batch 64 --epoch 15 --gamma 0.9
python train.py --batch 64 --epoch 15 --gamma 1.0


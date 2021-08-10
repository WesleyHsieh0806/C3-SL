Dump_path='./log/Lambda_5e-4_Log'
BATCH=64
EPOCH=70
LAMBDA=0.0005
python train_MSE.py --dump_path $Dump_path --batch $BATCH --epoch $EPOCH --Lambda $LAMBDA

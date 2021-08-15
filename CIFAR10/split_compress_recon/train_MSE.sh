Dump_path='./log/Lambda_0_Log'
BATCH=64
EPOCH=70
LAMBDA=0.0
python train_MSE.py --dump_path $Dump_path --batch $BATCH --epoch $EPOCH --Lambda $LAMBDA

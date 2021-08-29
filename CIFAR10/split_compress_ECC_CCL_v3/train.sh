DUMP_PATH='./log/Lambda_1e-1_Batch64_Ep100'
BATCH=64
EPOCH=100
Lambda=0.1
beta=0.005
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --beta $beta

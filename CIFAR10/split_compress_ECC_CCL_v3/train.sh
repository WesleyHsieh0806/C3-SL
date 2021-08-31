DUMP_PATH='./log/L0.1_B10.0_Batch64_Ep100'
BATCH=64
EPOCH=100
Lambda=0.1
beta=10.0
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --beta $beta

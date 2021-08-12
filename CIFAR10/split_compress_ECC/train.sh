BATCH=64
EPOCH=70
Lambda=0.0
DUMP_PATH='./log/Lambda_0_Batch64'
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH

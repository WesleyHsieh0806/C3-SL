DUMP_PATH='./log/Lambda_1.0_Batch64'
BATCH=64
EPOCH=70
Lambda=1.0
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH

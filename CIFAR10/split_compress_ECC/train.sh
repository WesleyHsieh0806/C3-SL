DUMP_PATH='./log/Lambda_5e-3_Batch64'
BATCH=64
EPOCH=70
Lambda=0.005
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH

DUMP_PATH='./log/Lambda_5e-2_Batch64'
BATCH=64
EPOCH=70
Lambda=0.05
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH

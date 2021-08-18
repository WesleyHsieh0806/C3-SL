DUMP_PATH='./log/Lambda_0_Batch64'
BATCH=64
EPOCH=70
Lambda=0.0
Restore_path='./log/Lambda_0_Batch64'
python collect_z.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --restore_path $Restore_path

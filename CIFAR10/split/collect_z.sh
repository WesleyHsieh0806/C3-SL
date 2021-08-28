DUMP_PATH='./log/Ep100'
BATCH=64
EPOCH=70
Lambda=0.0
Restore_path='./log/Ep100'
python collect_z.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --restore_path $Restore_path

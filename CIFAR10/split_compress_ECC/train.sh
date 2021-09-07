DUMP_PATH='./Resnet50_log/Scheduler/Lambda_5_Batch128_ep100'
BATCH=128
EPOCH=100
Lambda=5.0
ARCH="resnet50"
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --arch $ARCH

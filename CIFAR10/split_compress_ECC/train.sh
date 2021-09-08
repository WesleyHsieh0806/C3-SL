DUMP_PATH='./Resnet50_log/Scheduler/Lambda_0_Batch256_ep200'
BATCH=256
EPOCH=200
Lambda=0.0
ARCH="resnet50"
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --arch $ARCH

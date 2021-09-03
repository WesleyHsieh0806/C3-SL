DUMP_PATH='./log/Lambda_0_Batch64_ep100'
BATCH=64
EPOCH=100
Lambda=0.0
ARCH="resnet50"
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --arch $ARCH

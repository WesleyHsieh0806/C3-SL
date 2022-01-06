DUMP_PATH='./Resnet50_log/Early/Lambda_0_Batch64_ep100'
BATCH=64
EPOCH=100
Lambda=0.0
ARCH="resnet50"
SPLIT="early"
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --arch $ARCH --split $SPLIT

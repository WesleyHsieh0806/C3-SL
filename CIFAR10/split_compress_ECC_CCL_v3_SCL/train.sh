DUMP_PATH='./Resnet50_log/L0.1_B5.0_L1.0_Batch64_Ep100'
BATCH=64
EPOCH=100
Lambda=0.1
beta=5.0
Lambda2=1.0
ARCH="resnet50"
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --beta $beta --arch $ARCH --Lambda2 $Lambda2

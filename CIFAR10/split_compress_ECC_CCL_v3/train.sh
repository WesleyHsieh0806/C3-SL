DUMP_PATH='./Resnet50_log/Scheduler/L0.1_B5.0_Batch256_Ep200'
BATCH=256
EPOCH=200
Lambda=0.1
beta=5.0
ARCH="resnet50"
python train.py --batch $BATCH --epoch $EPOCH --Lambda $Lambda --dump_path $DUMP_PATH --beta $beta --arch $ARCH

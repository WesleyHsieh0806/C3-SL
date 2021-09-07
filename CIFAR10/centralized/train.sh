Dump_path='./Resnet50_log/Ep100'
EPOCH=100
ARCH="resnet50"
BATCH=64
python train.py --dump_path $Dump_path --epoch $EPOCH --arch $ARCH --batch $BATCH

Dump_path='./log'
EPOCH=70
ARCH="resnet50"
BATCH=64
python train.py --dump_path $Dump_path --epoch $EPOCH --arch $ARCH --batch $BATCH

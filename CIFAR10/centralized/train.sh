Dump_path='./log'
EPOCH=70
ARCH="resnet50"
python train.py --dump_path $Dump_path --epoch $EPOCH --arch $ARCH

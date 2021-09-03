Dump_path='./log/ResNet_Ep100'
BATCH=64
EPOCH=100
ARCH='resnet50'
python train.py --dump_path $Dump_path --batch $BATCH --epoch $EPOCH --arch $ARCH

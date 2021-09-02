Dump_path='./log/Ep100'
BATCH=64
EPOCH=100
ARCH='alexnet'
python train.py --dump_path $Dump_path --batch $BATCH --epoch $EPOCH --arch $ARCH

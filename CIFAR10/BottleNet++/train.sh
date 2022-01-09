BATCH=64
EPOCH=100
ARCH='vgg16'
SPLIT="middle-2"
Compression_ratio=4
Dump_path="./${ARCH}_log/${SPLIT}/B${BATCH}_Compress${Compression_ratio}_Ep${EPOCH}"
python3 train.py --dump_path $Dump_path --batch $BATCH --epoch $EPOCH --arch $ARCH --split $SPLIT --compression_ratio $Compression_ratio

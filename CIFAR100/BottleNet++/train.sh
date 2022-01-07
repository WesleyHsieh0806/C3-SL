BATCH=64
EPOCH=150
ARCH='resnet50'
SPLIT="middle-2"
Compression_ratio=4
Dump_path="./Resnet50_log/${SPLIT}/B${BATCH}_Compress${Compression_ratio}_Ep${EPOCH}"
python3 train.py --dump_path $Dump_path --batch $BATCH --epoch $EPOCH --arch $ARCH --split $SPLIT --compression_ratio $Compression_ratio

Dump_path='./Resnet50_log/Early/B64_Compress64_Ep200'
BATCH=64
EPOCH=200
ARCH='resnet50'
SPLIT="early"
Compression_ratio=64
python train.py --dump_path $Dump_path --batch $BATCH --epoch $EPOCH --arch $ARCH --split $SPLIT --compression_ratio $Compression_ratio

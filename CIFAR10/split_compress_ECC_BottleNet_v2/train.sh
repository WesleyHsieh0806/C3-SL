DUMP_PATH='./Resnet50_log/Middle/Compress64_Batch64_ep200'
BATCH=64
EPOCH=200
ARCH="resnet50"
SPLIT="middle"
WARMUP=1
Compression_ratio=64


python train.py \
--batch $BATCH \
--epoch $EPOCH \
--dump_path $DUMP_PATH \
--arch $ARCH \
--split $SPLIT \
--compression_ratio $Compression_ratio \
--warmup_epoch $WARMUP

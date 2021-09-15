BATCH=64
EPOCH=150
ARCH="resnet50"
SPLIT="middle"
Batch_Compression_Ratio=8
DUMP_PATH="./Resnet50_log/Middle/Compression${Batch_Compression_Ratio}_Batch${BATCH}_ep${EPOCH}"
python train.py \
--batch $BATCH \
--epoch $EPOCH \
--Lambda $Lambda \
--dump_path $DUMP_PATH \
--arch $ARCH \
--split $SPLIT \
--bcr $Batch_Compression_Ratio
BATCH=64
EPOCH=150
ARCH="resnet50"
SPLIT="middle-2"
Batch_Compression_Ratio=2
DUMP_PATH="./Resnet50_log/${SPLIT}/Compression${Batch_Compression_Ratio}_Batch${BATCH}_ep${EPOCH}"
python train.py \
--batch $BATCH \
--epoch $EPOCH \
--dump_path $DUMP_PATH \
--arch $ARCH \
--split $SPLIT \
--bcr $Batch_Compression_Ratio

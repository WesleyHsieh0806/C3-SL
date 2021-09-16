BATCH=64
EPOCH=200
ARCH="resnet50"
SPLIT="Middle"
PHASE1=20
PHASE2=30
Compression_ratio=64
Batch_Compression_Ratio=8
DUMP_PATH="./Resnet50_log/${SPLIT}/FC${Compression_ratio}_BC${Batch_Compression_Ratio}_Batch${BATCH}_ep${EPOCH}"

python train.py \
--batch $BATCH \
--epoch $EPOCH \
--dump_path $DUMP_PATH \
--arch $ARCH \
--split $SPLIT \
--compression_ratio $Compression_ratio \
--phase1_epoch $PHASE1 \
--phase2_epoch $PHASE2 \
--bcr $Batch_Compression_Ratio

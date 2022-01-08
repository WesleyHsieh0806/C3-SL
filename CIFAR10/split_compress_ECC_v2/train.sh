BATCH=64
EPOCH=100
ARCH="vgg16"
SPLIT="middle-2"
Batch_Compression_Ratio=2
DUMP_PATH="./${ARCH}_log/${SPLIT}/Compression${Batch_Compression_Ratio}_Batch${BATCH}_ep${EPOCH}"
python train.py \
--batch $BATCH \
--epoch $EPOCH \
--dump_path $DUMP_PATH \
--arch $ARCH \
--split $SPLIT \
--bcr $Batch_Compression_Ratio

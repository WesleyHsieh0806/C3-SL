BATCH=64
EPOCH=150
ARCH="resnet50"
SPLIT="middle-2"
<<<<<<< HEAD
Batch_Compression_Ratio=4
DUMP_PATH="./Resnet50_log/${SPLIT}/Compression${Batch_Compression_Ratio}_Batch${BATCH}_ep${EPOCH}"
python3 train.py \
=======
Batch_Compression_Ratio=2
DUMP_PATH="./Resnet50_log/${SPLIT}/Compression${Batch_Compression_Ratio}_Batch${BATCH}_ep${EPOCH}"
python train.py \
>>>>>>> 034f9feacf0cda0cb33552ba6ee6df1c86f30f1e
--batch $BATCH \
--epoch $EPOCH \
--dump_path $DUMP_PATH \
--arch $ARCH \
--split $SPLIT \
--bcr $Batch_Compression_Ratio

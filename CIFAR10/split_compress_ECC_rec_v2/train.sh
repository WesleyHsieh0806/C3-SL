# arguments
DUMP_PATH='./log/L50_Batch64_Ep100'
BATCH=64
EPOCH=100
Lambda=100.0

# Execute the training script
python train.py --batch $BATCH \
--epoch $EPOCH \
--Lambda $Lambda \
--dump_path $DUMP_PATH \

# arguments
DUMP_PATH='./log/L100_B0.005/Batch64_Ep100'
BATCH=64
EPOCH=100
Lambda=100.0
Lambda2=100.0
beta=0.005
beta2=0.005

# Execute the training script
python train.py --batch $BATCH \
--epoch $EPOCH \
--Lambda $Lambda \
--Lambda2 $Lambda2 \
--dump_path $DUMP_PATH \
--beta $beta \
--beta2 $beta2

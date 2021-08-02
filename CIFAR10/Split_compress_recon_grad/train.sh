BATCH=64
EPOCH=70
Lambda=0.0005
DUMP_PATH='./log/Lambda_5e-4_Batch64'
python train.py --batch ${batch size} --epoch ${epoch} --Lambda 0.0005 --dump_path ./log
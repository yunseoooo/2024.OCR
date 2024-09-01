CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
python tools/train_net.py --config-file configs/DPText_DETR/MPSC/R_50_poly.yaml \
 --num-gpus 4

# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
# python tools/train_net.py --config-file configs/DPText_DETR/MPSC/R_50_poly.yaml \
#  --num-gpus 4
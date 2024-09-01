CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \
python tools/train_net.py --config-file configs/Bridge/MPSC/R_50_poly.yaml \
 --num-gpus 1  MODEL.WEIGHTS ./adet/checkpoint/mpsc/model_final.pth


# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
# python tools/train_net.py --config-file configs/Bridge/TotalText/R_50_poly.yaml \
#  --num-gpus 1  MODEL.WEIGHTS ./adet/checkpoint/totaltext_final.pth

 
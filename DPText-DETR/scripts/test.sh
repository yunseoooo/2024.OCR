CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
python tools/train_net.py --config-file configs/DPText_DETR/MPSC/R_50_poly.yaml \
 --eval-only MODEL.WEIGHTS ./output/r_50_poly/MPSC/finetune/model_final.pth

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python demo/demo.py --config-file configs/DPText_DETR/MPSC/R_50_poly_rotate.yaml \
 --input ./datasets/MPSC/image/test_rotate/ --output ./vis/mpsc_rotate/ \
 --opts MODEL.WEIGHTS ./output/r_50_poly/MPSC/finetune/model_final.pth

#  python demo/demo.py --config-file configs/DPText_DETR/TotalText/R_50_poly.yaml \
#  --input ./datasets/totaltext/test_images_rotate/ --output ./vis/totaltext/ \
#  --opts MODEL.WEIGHTS ./adet/checkpoint/totaltext_final.pth

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python demo/demo.py --config-file configs/Bridge/MPSC/R_50_poly.yaml \
 --input ./datasets/MPSC/image/test/ --output ./vis/mpsc/real_lmdb_128batch/ \
 --opts MODEL.WEIGHTS ./output/Bridge/MPSC/finetune/real_lmdb_128batch/model_final.pth



# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
# python demo/demo.py --config-file configs/Bridge/TotalText/R_50_poly.yaml \
#  --input ./datasets/totaltext/test_images_rotate/ --output ./vis/totaltext/ \
#  --opts MODEL.WEIGHTS ./output/Bridge/TotalText/R_50_Polygon/model_final.pth
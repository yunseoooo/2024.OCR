CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python tools/train_net.py --config-file configs/Bridge/MPSC/R_50_poly.yaml \
 --num-gpus 1  --eval-only MODEL.WEIGHTS /home/ysjeong/workspace/OCR/Bridging_Spotting/Bridging-Text-Spotting/output/Bridge/MPSC/finetune/baseline/model_0004999.pth


# Set the path to save checkpoints
OUTPUT_DIR='output/pretrain_dig/'
# path to imagenet-1k set
DATA_PATH='/home/ysjeong/workspace/OCR/DiG/data/GoogleCC/lmdb'
# path to pretrain model

# Batch size can be adjusted according to the graphics card
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_mae_pretraining_moco.py \
 --image_alone_path ${DATA_PATH} \
 --mask_ratio 0.7 \
 --batch_size 128 \
 --opt adamw \
 --output_dir ${OUTPUT_DIR} \
 --epochs 10 \
 --warmup_steps 5000 \
 --max_len 25 \
 --num_view 2 \
 --moco_dim 256 \
 --moco_mlp_dim 4096 \
 --moco_m 0.99 \
 --moco_m_cos \
 --moco_t 0.2 \
 --num_windows 4 \
 --contrast_warmup_steps 0 \
 --contrast_start_epoch 0 \
 --loss_weight_pixel 1. \
 --loss_weight_contrast 0.1 \
 --only_mim_on_ori_img \
 --weight_decay 0.1 \
 --opt_betas 0.9 0.999 \
 --model pretrain_simmim_moco_ori_vit_small_patch4_32x128 \
 --patchnet_name no_patchtrans \
 --encoder_type vit
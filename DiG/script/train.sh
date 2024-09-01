# Set the path to save checkpoints
OUTPUT_DIR='output/DiG_/'
# path to imagenet-1k set
DATA_PATH='/home/ysjeong/workspace/OCR/DiG/data/instances/LMDB_train_'
# DATA_PATH='/home/ysjeong/workspace/OCR/DiG/data/instances/LMDB'
EVAL_DATA_PATH='/home/ysjeong/workspace/OCR/DiG/data/instances/LMDB_test_'
# path to pretrain model
MODEL_PATH='checkpoint/pretrain/checkpoint-9.pth'

# Batch size can be adjusted according to the graphics card
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 python run_class_finetuning.py \
    --model simmim_vit_small_patch4_32x128 \
    --data_path ${DATA_PATH} \
    --eval_data_path ${EVAL_DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --data_set image_lmdb \
    --nb_classes 97 \
    --smoothing 0.0 \
    --max_len 25 \
    --epochs 100 \
    --warmup_epochs 1 \
    --drop 0.1 \
    --attn_drop_rate 0.1 \
    --drop_path 0.1 \
    --dist_eval \
    --lr 1e-3 \
    --num_samples 1 \
    --fixed_encoder_layers 0 \
    --decoder_name tf_decoder \
    --use_abi_aug \
    --num_view 2 \
    --num_workers 8 \
    # --enable_deepspeed

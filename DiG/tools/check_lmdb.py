import lmdb
import os
from PIL import Image, ImageDraw, ImageFont
import six
import numpy as np
import cv2

def open_lmdb(lmdb_path, readonly=True):
    env = lmdb.open(lmdb_path, readonly=readonly, lock=False, readahead=False, meminit=False)
    return env

def read_lmdb(env):
    data = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data[key.decode('utf-8')] = value
    return data

def save_lmdb(data, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    with env.begin(write=True) as txn:
        for key, value in data.items():
            txn.put(key.encode('utf-8'), value)
        txn.put(b'num-samples', str(len(data) // 2).encode('utf-8'))  
    env.close()

def reindex_keys(image_keys, label_keys, data):
    new_data = {}
    index = 0
    for img_key, lbl_key in zip(sorted(image_keys), sorted(label_keys)):
        new_img_key = f'image-{index + 1:09d}'
        new_lbl_key = f'label-{index + 1:09d}'
        new_data[new_img_key] = data[img_key]
        new_data[new_lbl_key] = data[lbl_key]
        index += 1
    return new_data

def check_image_is_valid(image_bin):
    if image_bin is None:
        return False
    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
    if image_buf.size == 0:
        return False
    img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return False
    return True

def split_lmdb_data(lmdb_path, train_lmdb_path, test_lmdb_path, num_test_samples=2042):
    env = open_lmdb(lmdb_path)
    data = read_lmdb(env)

    # Split data into train and test
    image_keys = [key for key in data.keys() if key.startswith('image-')]
    label_keys = [key for key in data.keys() if key.startswith('label-')]

    if len(image_keys) != len(label_keys):
        print("Error: Number of image keys and label keys do not match!")
        return

    # Split train and test data
    test_image_keys = image_keys[:num_test_samples]
    test_label_keys = label_keys[:num_test_samples]

    train_image_keys = image_keys[num_test_samples:12062]
    train_label_keys = label_keys[num_test_samples:12062]

    test_data = reindex_keys(test_image_keys, test_label_keys, data)
    train_data = reindex_keys(train_image_keys, train_label_keys, data)

    save_lmdb(test_data, test_lmdb_path)
    save_lmdb(train_data, train_lmdb_path)

    print(f"Train LMDB 데이터 수: {len(train_image_keys)}")
    print(f"Test LMDB 데이터 수: {len(test_image_keys)}")

def save_image_with_label(env, index, save_path):
    with env.begin() as txn:
        img_key = b'image-%09d' % (index + 1)
        label_key = b'label-%09d' % (index + 1)
        imgbuf = txn.get(img_key)
        label = txn.get(label_key).decode('utf-8')

        if imgbuf is None:
            print(f"Image at index {index} not found in LMDB.")
            return

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('RGB')
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((10, 10), label, font=font, fill=(255, 255, 255))

            img.save(save_path)
            print(f"Image at index {index} with label saved to {save_path}.")
        except IOError:
            print(f"Corrupted image for index {index} in LMDB.")

lmdb_path = '/home/ysjeong/workspace/OCR/DiG/data/instances/LMDB'
train_lmdb_path = './LMDB_train_'
test_lmdb_path = './LMDB_test_'

split_lmdb_data(lmdb_path, train_lmdb_path, test_lmdb_path)

train_env = open_lmdb(train_lmdb_path)
test_env = open_lmdb(test_lmdb_path)
save_image_with_label(train_env, 12062, 'train_12061_image.jpg')  
save_image_with_label(test_env, 2041, 'test_2041_image.jpg')  



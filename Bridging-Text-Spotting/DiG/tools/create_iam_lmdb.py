import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm
import six
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import re

def checkImageIsValid(imageBin):
  if imageBin is None:
    return False
  imageBuf = np.fromstring(imageBin, dtype=np.uint8)
  # imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
  if imageBuf.size == 0:
    return False
  img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
  imgH, imgW = img.shape[0], img.shape[1]
  if imgH * imgW == 0:
    return False
  return True


def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      txn.put(k.encode(), v)


def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
  """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath    : LMDB output path
      imagePathList : list of image path
      labelList     : list of corresponding groundtruth texts
      lexiconList   : (optional) list of lexicon lists
      checkValid    : if true, check the validity of every image
  """
  assert(len(imagePathList) == len(labelList))
  nSamples = len(imagePathList)
  env = lmdb.open(outputPath, map_size=1099511627776)
  cache = {}
  cnt = 1
  for i in tqdm(range(nSamples)):
    imagePath = imagePathList[i]
    label = labelList[i]
    if len(label) == 0:
      continue
    if not os.path.exists(imagePath):
      print('%s does not exist' % imagePath)
      continue
    with open(imagePath, 'rb') as f:
      imageBin = f.read()
    if checkValid:
      if not checkImageIsValid(imageBin):
        print('%s is not a valid image' % imagePath)
        continue

    imageKey = 'image-%09d' % cnt
    labelKey = 'label-%09d' % cnt
    cache[imageKey] = imageBin
    cache[labelKey] = label.encode()
    if lexiconList:
      lexiconKey = 'lexicon-%09d' % cnt
      cache[lexiconKey] = ' '.join(lexiconList[i])
    if cnt % 1000 == 0:
      writeCache(env, cache)
      cache = {}
      print('Written %d / %d' % (cnt, nSamples))
    cnt += 1
  nSamples = cnt-1
  cache['num-samples'] = str(nSamples).encode()
  writeCache(env, cache)
  print('Created dataset with %d samples' % nSamples)

if __name__ == "__main__":
  image_root_dir = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten/IAM/words'
  annot_path = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten/IAM/ascii/words.txt'
  lmdb_output_path = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten_lmdbs/IAM_train'
  split_path = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten/IAM/splits/trainset.txt'

  lmdb_output_path = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten_lmdbs/IAM_test'
  split_path = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten/IAM/splits/testset.txt'

  lmdb_output_path = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten_lmdbs/IAM_val1'
  split_path = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten/IAM/splits/validationset1.txt'

  lmdb_output_path = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten_lmdbs/IAM_val2'
  split_path = '/home/ymk-wh/workspace/datasets/text_recognition/handwritten/IAM/splits/validationset2.txt'

  # load annotation
  with open(annot_path, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
  # load data split
  with open(split_path, 'r') as f:
    split_ids = f.readlines()
    split_ids = [split_id.strip() for split_id in split_ids]
  
  image_path_list, label_list = [], []
  for line in lines:
    if line[0] == '#':
      continue
    splits = line.split(' ', 8)
    image_name, seg_flag, _, x, y, w, h, tag, label = splits
    if seg_flag == 'ok':
      ids = image_name.split('-')
      paper_id = ids[0]
      line_id = '-'.join(ids[:2])
      split_id = '-'.join(ids[:3])
      if split_id in split_ids:
        image_path = os.path.join(image_root_dir, paper_id, line_id, image_name + '.png')
        image_path_list.append(image_path)
        label_list.append(label)
    
  createDataset(lmdb_output_path, image_path_list, label_list)
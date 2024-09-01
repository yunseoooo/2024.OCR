'''
# txt, jpg 0부터 이름 새로 매핑용
import os

folder_path = '/home/ysjeong/workspace/OCR/DPText-DETR/datasets/MPSC/annotation/rotate_test' 

file_names = os.listdir(folder_path)
file_names.sort(key=lambda x: int(x.split('_')[2].split('.')[0])) 

for i, file_name in enumerate(file_names):
    new_name = f"gt_img_{i}.txt"
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path) 

    print(f"Renamed '{file_name}' to '{new_name}'")  
'''

import json

file_path = '/home/ysjeong/workspace/OCR/DPText-DETR/datasets/MPSC/annotation/rotate_test.json'

with open(file_path, 'r') as file:
    data = json.load(file)

for i, anno in enumerate(data['annotations']):

    anno['image_id'] -= 1

# image_id_map = {image['id']: i for i, image in enumerate(data['images'])}
# for annotation in data['annotations']:
#     annotation['image_id'] = image_id_map[annotation['image_id']]

with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)



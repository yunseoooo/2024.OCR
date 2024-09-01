import json
import os
import re
from PIL import Image
import numpy as np

def interpolate_points(points, num_points):
    points = np.array(points).reshape(-1, 2)
    num_existing_points = len(points)
    new_points = []
    for i in range(num_existing_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_existing_points]
        new_points.append(p1)
        for j in range(1, num_points // num_existing_points):
            new_points.append(p1 + j * (p2 - p1) / (num_points // num_existing_points))
    return np.array(new_points).flatten().tolist()

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
                '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@',
                'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
                'R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b',
                'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
                't','u','v','w','x','y','z','{','|','}','~']

def decode_text_to_rec(text, max_length=25):
    rec = [CTLABELS.index(char) if char in CTLABELS else 96 for char in text]
    rec += [96] * (max_length - len(rec))  # Fill remaining length with 96
    return rec[:max_length]

image_dir = "datasets/MPSC/image/train/"
annotation_dir = "datasets/MPSC/annotation/train/"
json_file = "datasets/MPSC/annotation/train.json"

categories = [{"id": 1, "name": "text"}]

images = []
annotations = []

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
image_files.sort(key=natural_sort_key)

for image_id, image_file in enumerate(image_files):
    with Image.open(os.path.join(image_dir, image_file)) as img:
        width, height = img.size
    images.append({"id": image_id, "file_name": image_file, "width": width, "height": height})
    
    txt_file = 'gt_img_' + image_file.split('_')[2].replace('.jpg', '.txt')
    txt_path = os.path.join(annotation_dir, txt_file)
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                parts = re.split(r'[,\n]', line.strip())
                try:
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                    text = ','.join(parts[8:])
                    if text == "###":
                        continue
                except ValueError as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue

                bbox = [min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4) - min(x1, x2, x3, x4), max(y1, y2, y3, y4) - min(y1, y2, y3, y4)]
                area = bbox[2] * bbox[3]
                polys = [x1, y1, x2, y2, x3, y3, x4, y4]

                if len(polys) == 8:
                    polys = interpolate_points(polys, 16)
                else:
                    print(f"Invalid number of polygon points in {txt_file}: {line}")
                    continue

                rec = decode_text_to_rec(text)

                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "polys": polys,
                    "rec": rec
                })

coco_format = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(json_file, 'w') as f:
    json.dump(coco_format, f, indent=4)

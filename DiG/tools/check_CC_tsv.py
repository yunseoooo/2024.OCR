import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# 파일 경로 지정
tsv_file = '/data2/datasets/OCR/CC_OCR/Train_GCC-training.tsv'

# TSV 파일 읽기
data = pd.read_csv(tsv_file, sep='\t')

image_url = data.iloc[30, 1]  

response = requests.get(image_url)
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    
    output_image_path = "./downloaded_image.jpg"
    image.save(output_image_path)
    print(f"Image saved to {output_image_path}")
else:
    print(f"Failed to download image. Status code: {response.status_code}")

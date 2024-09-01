import lmdb
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

def load_ocr_boxes(file_path):
    """Load OCR boxes from .npy file."""
    data = np.load(file_path, allow_pickle=True, encoding='latin1')
    return data.item()

def download_image(url):
    """Download image from URL and return PIL Image object."""
    try:
        response = requests.get(url, timeout=10)  # Added timeout
        response.raise_for_status()  # Check if request was successful
        return Image.open(BytesIO(response.content))
    except requests.HTTPError as e:
        print(f"HTTP Error downloading image from {url}: {e}")
    except requests.RequestException as e:
        print(f"Request Error downloading image from {url}: {e}")
    except IOError as e:
        print(f"IO Error opening image from response content: {e}")
    return None

def crop_text_boxes(image, ocr_boxes):
    """Crop the image based on OCR boxes and return as byte data."""
    cropped_images = []
    for box in ocr_boxes['ocr_boxes']:
        x_min, y_min, x_max, y_max = box
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        buffered = BytesIO()
        cropped_image.save(buffered, format="JPEG")
        cropped_images.append(buffered.getvalue())
    return cropped_images

def process_and_store_images(tsv_file_path, ocr_dir_path, lmdb_path):
    """Process images and store cropped text boxes into LMDB."""
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs=10)
    
    try:
        with env.begin(write=True) as txn:
            # Load TSV file
            data = pd.read_csv(tsv_file_path, sep='\t', header=None, names=['description', 'url'])
            total_files = len(data)
            num_samples = 0
            
            # Iterate over OCR files in the directory
            for i, ocr_file in enumerate(os.listdir(ocr_dir_path)):
                if not ocr_file.endswith('_info.npy'):
                    continue
                
                try:
                    # Extract file index from the OCR file name
                    file_index = int(ocr_file.split('_')[0])
                    
                    # Get the image URL from TSV file using file_index
                    if file_index >= total_files:
                        print(f"File index {file_index} is out of range for TSV data.")
                        continue
                    
                    row = data.iloc[file_index]
                    image_url = row['url']
                    
                    # Construct the full path to the OCR file
                    ocr_file_path = os.path.join(ocr_dir_path, ocr_file)
                    
                    # Download the image
                    image = download_image(image_url)
                    if image is None:
                        continue
                    
                    # Load OCR boxes
                    try:
                        ocr_boxes = load_ocr_boxes(ocr_file_path)
                    except FileNotFoundError:
                        print(f"Error processing file index {file_index}: OCR file not found: {ocr_file_path}")
                        continue
                    
                    # Crop text boxes and store in LMDB
                    cropped_images = crop_text_boxes(image, ocr_boxes)
                    
                    for j, cropped_image_data in enumerate(cropped_images):
                        key = f'image-{file_index:09d}-{j:03d}'.encode('ascii')
                        txn.put(key, cropped_image_data)
                    
                    num_samples += 1
                    
                    # Print progress
                    if i % 100 == 0 or i == total_files - 1:
                        progress = (i + 1) / total_files * 100
                        print(f'Processed {i + 1} of {total_files} images ({progress:.2f}%)')
                
                except Exception as e:
                    print(f"Error processing OCR file {ocr_file}: {e}")
            
            txn.put(b'num-samples', str(num_samples).encode('ascii'))
    finally:
        env.close()

    print(f'LMDB created at {lmdb_path}')

tsv_file_path = '/data2/datasets/OCR/CC_OCR/GoogleCC/Train_GCC-training.tsv'
ocr_dir_path = '/data2/datasets/OCR/CC_OCR/GoogleCC/ocr_feat/training'
lmdb_path = '/data2/datasets/OCR/CC_OCR/GoogleCC/lmdb'

process_and_store_images(tsv_file_path, ocr_dir_path, lmdb_path)

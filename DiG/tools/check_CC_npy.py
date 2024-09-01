# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from PIL import Image

# directory = '/data2/datasets/OCR/CC_OCR/GoogleCC/ocr_feat/training'

# files = os.listdir(directory)

# example_file = os.path.join(directory, files[0])

# data = np.load(example_file, allow_pickle=True, encoding='latin1')

# print("Data shape:", data.shape)
# print("Data type:", data.dtype)

# breakpoint()
# image = Image.fromarray(data)

# output_image_path = "./example.jpg"
# image.save(output_image_path)

# print(f"Image saved to {output_image_path}")


import numpy as np
import os

directory = '/data2/datasets/OCR/CC_OCR/GoogleCC/ocr_feat/training'
breakpoint()

base_filename = '1322621_4231405636'

npy_file = os.path.join(directory, base_filename + '.npy')
info_file = os.path.join(directory, base_filename + '_info.npy')

npy_data = np.load(npy_file, allow_pickle=True)
info_data = np.load(info_file, allow_pickle=True, encoding='latin1')

breakpoint()

print(f"Data from {npy_file}:")
print(npy_data)
print("\n")

print(f"Data from {info_file}:")
print(info_data)

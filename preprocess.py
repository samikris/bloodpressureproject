import h5py
import numpy as np
from PIL import Image # Or import cv2
import os

hdf5_file_path = 'image_dataset_2.h5'
image_folder = '/Users/samikris/downloads/Images'

with h5py.File(hdf5_file_path, 'w') as hf:
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Pre-allocate a dataset if you know the total number of images and their dimensions
    # For example, if all images are 256x256 RGB:
    # num_images = len(image_files)
    # images_dset = hf.create_dataset('images', (num_images, IMG_HEIGHT, IMG_WIDTH, 3), dtype='uint8')

    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path)
            # If resizing is needed:
            # img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img)

            # Store the image array in the HDF5 dataset
            # If pre-allocated: images_dset[i] = img_array
            # Otherwise, create a new dataset for each image:
            hf.create_dataset(img_path, data=img_array)

            # You can also store metadata, like labels or original filenames
            # hf.create_dataset(f'label_{i}', data='some_label')
        except Exception as e:
            print(f"Error processing {img_path}: {e}")






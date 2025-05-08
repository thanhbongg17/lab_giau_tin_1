# Sinh ảnh watermark bằng dịch hình học
################################################################
import cv2
import numpy as np
import os
import random
from shutil import copyfile

def embed_geo_watermark(img, tx=3, ty=3):
    """Dịch ảnh để tạo watermark hình học"""
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted

# Thư mục gốc và đích
src_dir = "imgs"
base_out = "datasets/geo_wm"
train_ratio = 0.8

# Tạo các thư mục output
for subfolder in ["trainA", "trainB", "testA", "testB"]:
    os.makedirs(os.path.join(base_out, subfolder), exist_ok=True)

# Lấy danh sách và xáo trộn ảnh
all_images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(all_images)

# Tính ngưỡng chia train/test
split_index = int(len(all_images) * train_ratio)

# Chia và xử lý
for idx, filename in enumerate(all_images):
    img_path = os.path.join(src_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    wm_img = embed_geo_watermark(img)

    out_name = f"{idx:04d}.png"

    if idx < split_index:
        # Train
        cv2.imwrite(os.path.join(base_out, "trainA", out_name), wm_img)
        cv2.imwrite(os.path.join(base_out, "trainB", out_name), img)
    else:
        # Test
        cv2.imwrite(os.path.join(base_out, "testA", out_name), wm_img)
        cv2.imwrite(os.path.join(base_out, "testB", out_name), img)

print("✅ Đã chia ảnh thành trainA/trainB và testA/testB thành công!")

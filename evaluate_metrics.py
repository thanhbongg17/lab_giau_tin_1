from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.exposure import match_histograms
from PIL import Image
import cv2
import numpy as np
import os

# Tạo thư mục lưu kết quả nếu chưa có
os.makedirs("results", exist_ok=True)

# 1. Load ảnh gốc (grayscale)
img1 = cv2.imread("imgs/edges2cats.jpg", cv2.IMREAD_GRAYSCALE)

# 2. Load ảnh gif (frame đầu), resize theo kích thước ảnh gốc và chuyển sang grayscale
gif = Image.open("imgs/horse2zebra.gif")
gif_resized = gif.resize((img1.shape[1], img1.shape[0]), Image.LANCZOS)
img2_rgb = np.array(gif_resized.convert("RGB"))
img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

# 3. Tăng sáng và tương phản nhẹ
img2 = cv2.convertScaleAbs(img2, alpha=1.3, beta=25)  # Tăng alpha và beta cho sự thay đổi rõ nét hơn

# 4. So khớp histogram với ảnh gốc
img2 = match_histograms(img2, img1, channel_axis=None).astype(np.uint8)

# 5. Tăng cường tương phản cục bộ bằng CLAHE
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # Tăng clipLimit để cải thiện độ tương phản
img2 = clahe.apply(img2)

# 6. Làm sắc nét bằng kỹ thuật unsharp mask
blurred = cv2.GaussianBlur(img2, (5, 5), 1.0)
img2 = cv2.addWeighted(img2, 1.5, blurred, -0.5, 0)

# 7. Lọc Bilateral để bảo toàn cạnh và giảm nhiễu
img2 = cv2.bilateralFilter(img2, d=20, sigmaColor=75, sigmaSpace=75)  # Tăng d để bảo toàn cạnh tốt hơn

# 8. Khử nhiễu nhẹ bằng Non-local Means
img2 = cv2.fastNlMeansDenoising(img2, None, h=20, templateWindowSize=7, searchWindowSize=21)

# 9. Tính SSIM và PSNR giữa ảnh gốc và ảnh khôi phục
ssim_value = ssim(img1, img2)
psnr_value = psnr(img1, img2)

# 10. In kết quả
print("SSIM:", ssim_value)
print("PSNR:", psnr_value)

# 11. Hiển thị ảnh
cv2.imshow("Original (img1)", img1)
cv2.imshow("Recovered (img2)", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 12. Lưu ảnh kết quả để chèn vào báo cáo
cv2.imwrite("results/recovered_image.png", img2)

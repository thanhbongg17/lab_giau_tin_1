import os
import sys
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import tensor2im

if __name__ == '__main__':
    # Auto-insert --dataroot nếu bạn không truyền
    if '--dataroot' not in sys.argv:
        sys.argv.extend(['--dataroot', './datasets/geo_wm'])
        print("[Info] --dataroot not provided, auto-using './datasets/geo_wm'")

    # 1. Parse test options
    opt = TestOptions().parse()
    opt.phase = 'test'
    opt.epoch = 'latest'
    opt.load_iter = 0
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # 2. Create dataset & model
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # 3. Load and fix clean_model.pth
    clean_path = 'clean_model.pth'
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"{clean_path} not found. Run clean_model.py first.")
    raw_state = torch.load(clean_path, map_location=model.device)

    fixed_state = {}
for key, tensor in raw_state.items():
    # Bỏ qua weight_mask
    if 'weight_mask' in key:
        continue
    # Chuyển weight_orig -> weight
    elif 'weight_orig' in key:
        fixed_key = key.replace('weight_orig', 'weight')
        fixed_state[fixed_key] = tensor
    else:
        fixed_state[key] = tensor  # bias và các key khác giữ nguyên


    model.netG_A.load_state_dict(fixed_state, strict=False)
    model.netG_A.eval()

    # 4. Prepare HTML output dir
    web_dir = os.path.join(opt.results_dir, opt.name, 'clean_latest')
    print(f"[Info] Saving cleaned outputs to: {web_dir}")
    webpage = html.HTML(web_dir, f'Clean verification: {opt.name}')

    # 5. Inference + compute SSIM
    ssim_vals = []
    for data in dataset:
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        img_A   = tensor2im(visuals['real_A'])
        img_rec = tensor2im(visuals['fake_B'])
        g1 = np.mean(img_A, axis=2)
        g2 = np.mean(img_rec, axis=2)
        s = ssim(g1, g2, data_range=255.0)
        ssim_vals.append(s)

        save_images(webpage, visuals, model.get_image_paths())
    webpage.save()

    # 6. Report and detect
    avg_ssim = float(np.mean(ssim_vals))
    print(f"Accuracy (avg SSIM) after cleaning: {avg_ssim:.4f}")
    if avg_ssim > 0.90:
        print("Watermark not detected")
    else:
        print("Watermark detected")

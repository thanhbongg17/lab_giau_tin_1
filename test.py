import os
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import tensor2im  # <-- cần thiết để chuyển tensor thành numpy
from skimage.metrics import structural_similarity as ssim  # <-- Thêm SSIM

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)      # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)          # create a model given opt.model and other options
    model.setup(opt)                   # load networks; create schedulers

    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    if opt.load_iter > 0:
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        # ==== THÊM PHẦN PHÁT HIỆN BẤT THƯỜNG ==== #
        input_image = visuals.get('real_A')
        output_image = visuals.get('fake_B')

        if input_image is not None and output_image is not None:
            input_np = tensor2im(input_image)
            output_np = tensor2im(output_image)

            input_gray = np.mean(input_np, axis=2)
            output_gray = np.mean(output_np, axis=2)

            similarity = ssim(input_gray, output_gray, data_range=255.0)


            if similarity > 0.95:
                print("Prediction confidence unusually high")
        # ========================================= #

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize, use_wandb=opt.use_wandb)

    webpage.save()

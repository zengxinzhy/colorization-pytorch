import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image

import string
import torch
import torchvision
import torchvision.transforms as transforms


from util import util
import numpy as np

opt = TrainOptions().gather_options()
opt.isTrain = True
opt.name = "siggraph_caffemodel"
opt.mask_cent = 0
opt.gpu_ids = []
opt.load_model = True
opt.num_threads = 1   # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.display_id = -1  # no visdom display
opt.phase = 'val'
opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
opt.serial_batches = True
opt.aspect_ratio = 1.

# process opt.suffix
if opt.suffix:
    suffix = ('_' + opt.suffix.format(**vars(opt))
              ) if opt.suffix != '' else ''
    opt.name = opt.name + suffix

opt.A = 2 * opt.ab_max / opt.ab_quant + 1
opt.B = opt.A

model = create_model(opt)
model.setup(opt)
model.eval()

image_path = "./image.JPEG"
image = Image.open(image_path)
image.show(command='fim')
image = transforms.ToTensor()(image)
image = image.view(1, *image.shape)
image = util.crop_mult(image, mult=8)

data = util.get_colorization_data(
    [image], opt, ab_thresh=0., p=0.125)

model.set_input(data)
model.test(True)

to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr',
                'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]

visuals = util.get_subset_dict(
    model.get_current_visuals(), to_visualize)

for key, value in visuals.items():
    print(key, value.shape)
    transforms.ToPILImage()(value[0]).show(title=key, command='fim')

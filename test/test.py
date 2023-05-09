import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import os
import numpy as np
import torch
import cv2

from util import flow_util

opt = TestOptions().parse() # Parse command line arguments using TestOptions class

# Initialize starting epoch and iteration
start_epoch, epoch_iter = 1, 0

# Convert optical flow to color using flow2color function
f2c = flow_util.flow2color()

# Load test dataset using CreateDataLoader class
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()  # <class 'torch.utils.data.dataloader.DataLoader'>
dataset_size = len(data_loader)
print(dataset_size)

# Initialize AFWM model
warp_model = AFWM(opt, 3)
print(warp_model)
warp_model.eval()
warp_model.cuda()

# Load the checkpoint for AFWM model
load_checkpoint(warp_model, opt.warp_checkpoint)


# Initialize generator model and print it to console
gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, opt.gen_checkpoint)

total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize

if not os.path.exists('our_t_results'):
  os.mkdir('our_t_results')

for epoch in range(1,2):
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # data is dictionary
        # ['image', 'clothes', 'edge', 'p_name']

        real_image = data['image']
        clothes = data['clothes']
        ##edge is extracted from the clothes image with the built-in function in python
        edge = data['edge']
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
        clothes = clothes * edge        

        flow_out = warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        path = 'results/' + opt.name
        os.makedirs(path, exist_ok=True)
        print(data['p_name'])
        print(type(data['edge']))    # <class 'torch.Tensor'>


        if step % 1 == 0:
            
            ## save try-on image only

            utils.save_image(
                p_tryon,
                os.path.join('./our_t_results', data['p_name'][0]),
                nrow=int(1),
                normalize=True,
                value_range=(-1,1),
            )
        step += 1
        if epoch_iter >= dataset_size:
            break



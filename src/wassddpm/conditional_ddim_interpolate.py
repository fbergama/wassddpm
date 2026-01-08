###################################################################################
# WASSDDPM: Denoising Diffusion Probabilistic Models for scattered point cloud    #
#           interpolation of sea waves elevation data                             #
# Copyright (C) 2026 Ca' Foscari University of Venice                             #
#                                                                                 #
# This program is free software: you can redistribute it and/or modify it under   #
# the terms of the GNU General Public License as published by the Free Software   #
# Foundation, either version 3 of the License, or (at your option) any later      #
# version.                                                                        #
#                                                                                 #
# This program is distributed in the hope that it will be useful, but WITHOUT ANY #
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A #
# PARTICULAR PURPOSE. See the GNU General Public License for more details.        #
#                                                                                 #
# You should have received a copy of the GNU General Public License along         #
# with this program. If not, see <https://www.gnu.org/licenses/>.                 #
###################################################################################
#
#  Author(s):
#  - Shambel Fente Mengistu
#  - Filippo Bergamasco 
#
###################################################################################

from diffusers import DDPMScheduler, UNet2DModel
from diffusers import ConsistencyModelPipeline as CMP
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from accelerate import Accelerator
from accelerate import notebook_launcher
from huggingface_hub import create_repo, upload_folder
from accelerate import notebook_launcher
import tqdm
import glob
import scipy as sp
import cv2
from pathlib import Path
from diffusers.utils import make_image_grid, load_image, export_to_video
from diffusers import RePaintPipeline, RePaintScheduler
from torchvision.utils import make_grid
import os
import re
import torch
torch.cuda.empty_cache()
import torchvision.transforms.functional as fn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datasets import load_dataset
import cvbase as cvb
import h5py
import io
import sys

from .dataloader import load_data
from torch.utils.data import DataLoader


def conditional_DDIM_interpolate( h5_dataset,
                                  out_dir,
                                  gen_sample = 16,
                                  num_timesteps = 80,
                                  BATCH_SIZE = 32,
                                  max_batches = -1,
                                  debug_diffusion_process = False):


    print(f"Using {out_dir} to store interpolated outputs")
    if not os.path.isdir( out_dir ):
        print("Error: not a directory")
        return

    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # for sorting the images in ascending order in a file
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts


    # loading pretrained model and scheduler

    # reduced version Unet model
    script_path = os.path.dirname(os.path.realpath(__file__))

    model = UNet2DModel.from_pretrained(os.path.join(script_path,"Conditional_DDIM/unet"), use_safetensors=True).to("cuda")
    scheduler = DDPMScheduler.from_pretrained(os.path.join(script_path,"Conditional_DDIM/scheduler"))
    model.eval()     
    scheduler.set_timesteps(num_timesteps)


    data_set = load_data( h5_dataset )
    data_loader = DataLoader( data_set, batch_size = BATCH_SIZE, pin_memory=True, shuffle=False )

    #torch.manual_seed(42)
    batch_idx = 0
    img_idx = 0

    # sample shape
    sample_size  = model.config.sample_size
    # empty array for generated images (each sparse input 10 times with different initial noise)

    print("Starting")
    print("----------------")
    print(f" N samples: {gen_sample}")
    print(f"  Batch sz: {BATCH_SIZE}")
    print(f" timesteps: {num_timesteps}")
    print("----------------")

    for step, batch in enumerate(tqdm.tqdm(data_loader, desc="    batch", unit="batch")):
        images = batch[0].to('cuda')
        masks = batch[1].to('cuda')
        #images = masks*images


        batch_generated = np.zeros((images.shape[0], gen_sample, 256, 256))

        if max_batches>=0 and step>=max_batches:
            print("Max number of batches reached, exiting")
            sys.exit(0)


        # Generating gen_sample number of images
        for gs in tqdm.trange(gen_sample, desc="   sample", unit="sample", leave=False ):

            #print("\r["+("#"*int(gs))+"-"*(int(gen_sample)-int(gs)-1)+"]",end="",flush=True)
            
            # create some random noise with the same shape as the model output
            noise = torch.randn((images.shape[0],images.shape[1],sample_size, sample_size), device='cuda')
        
            # stacking the noise, images and mask for feeding to Unet
            model_input = torch.cat((noise, images, masks), dim=1)
      
            # Generate images using the diffusion model
            
            for t in tqdm.tqdm( scheduler.timesteps, desc="denoise t", unit="step", leave=False ):
                with torch.no_grad():

                    if debug_diffusion_process:
                        plt.figure()
                        ZZ = model_input[0, 0, :, :].cpu().numpy()
                        plt.imshow( ZZ )
                        plt.tight_layout()
                        plt.colorbar()
                        plt.savefig("generated_diffusion_process/diffusion_process_%06d.png"%t)
                        plt.close()


                    noisy_residual = model(model_input, t).sample
                    previous_noisy_sample = scheduler.step(noisy_residual[:,0].unsqueeze(1), t, noise).prev_sample   # loss is computed on the first channels of the predicted noise
                    noise = previous_noisy_sample
                    model_input = torch.cat((noise, images, masks), dim=1)
        

            if debug_diffusion_process:
                quit()
        
            img_idx = batch_idx
            
            # saving batch of generated images
            for i in range(images.shape[0]):

                img = (noise[i] / 2 + 0.5).clamp(0, 1)
                #img = (img* 255).round().to(torch.uint8).squeeze(0).cpu().numpy()
                
                img = img.squeeze(0).squeeze(0).cpu().numpy()
                batch_generated[i, gs,:,:] = img
                #img_idx +=1  
                #image_name = all_images[img_idx]
                #cv2.imwrite(os.path.join('/home/shambel/Diffusion/Experiment_new/steps_250' , image_name.replace('.png', '') + '_generated'+str(gs)+'.png'), img)
                '''
                if 'Percent_0.1%' in image_name:
                    cv2.imwrite(os.path.join('/home/shambel/Diffusion/Experiment', 'Percent_0.1' , image_name.replace('.png', '') + '_generated'+str(gs)+'.png'), img)
                elif 'Percent_0.5%' in image_name:
                    cv2.imwrite(os.path.join('/home/shambel/Diffusion/Experiment', 'Percent_0.5' , image_name.replace('.png', '') + '_generated'+str(gs)+'.png'), img)
                elif 'Percent_1%' in image_name:
                    cv2.imwrite(os.path.join('/home/shambel/Diffusion/Experiment', 'Percent_1' , image_name.replace('.png', '') + '_generated'+str(gs)+'.png'), img)
                elif 'Percent_3%' in image_name:
                    cv2.imwrite(os.path.join('/home/shambel/Diffusion/Experiment', 'Percent_3' , image_name.replace('.png', '') + '_generated'+str(gs)+'.png'), img)
                elif 'Percent_5%' in image_name:
                    cv2.imwrite(os.path.join('/home/shambel/Diffusion/Experiment', 'Percent_5', image_name.replace('.png', '') + '_generated'+str(gs)+'.png'), img)
                else:
                    pass   
                '''
        
            del model_input
            

        for i in range(images.shape[0]): 
            file_name = 'gen_image_%06d.h5'%img_idx
            path = os.path.join(out_dir, file_name)
            with h5py.File(path, 'w') as f:
                # Write the numpy array to the HDF5 file
                f.create_dataset('gen_wave', data=batch_generated[i])
            img_idx +=1 
            
        batch_idx +=images.shape[0]
        del noise, img
        torch.cuda.empty_cache()
        
       

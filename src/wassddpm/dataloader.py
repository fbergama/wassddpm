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
#
###################################################################################
import os
import torch
import numpy as np
import torch.utils.data as data
import h5py

class load_data(data.Dataset):

    def __init__(self, 
                 h5file,
                 transform = None,
                 mode = 'train'
                 ):
        self.mode = mode
        self.transform = transform
        self.h5f = h5file
        self.data = self.h5f['/GPM']
        self.zmin, self.zmax = self.data.attrs["zminmax"]

    
    def __len__(self):
        num_entries = self.h5f['/GPM'].shape[0]
        return num_entries
    
    def __getitem__(self, IDX):
        inputs = []
        
        #G = (((data[IDX,:,:,0] - zmin) / (zmax-zmin))*2)-1
        M = np.array( self.data[IDX,:,:,2] )
        P = ((((self.data[IDX,:,:,1] - self.zmin) / (self.zmax-self.zmin))*2)-1) * M
   
        return torch.tensor(P.astype(np.float32)).unsqueeze(0), torch.tensor(M.astype(np.float32)).unsqueeze(0)


# -*- coding: utf-8 -*-
#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# Evan Kiely (WMG)

from cil.io import ZEISSDataReader, NikonDataReader,TIFFWriter, TIFFStackReader
from cil.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData, BlockDataContainer
from readers.TescanDataReader import TescanDataReader
from readers.WaygateDataReader import WaygateDataReader


# CIL Processors
from cil.processors import CentreOfRotationCorrector, Slicer, TransmissionAbsorptionConverter, Normaliser, Padder, Binner, Masker, MaskGenerator, AbsorptionTransmissionConverter

# CIL display tools
from cil.utilities.display import show2D, show_geometry


# CIL standard reconstruction
from cil.recon import FBP, FDK

# CIL Optimisers/Iterative recon
from cil.optimisation.functions import  IndicatorBox, MixedL21Norm, L2NormSquared, BlockFunction, L1Norm, LeastSquares, OperatorCompositionFunction, TotalVariation, ZeroFunction, \
                                         KullbackLeibler, SmoothMixedL21Norm, MixedL11Norm
from cil.optimisation.algorithms import CGLS, SIRT, GD, FISTA, PDHG, SPDHG, ISTA, LADMM
from cil.optimisation.operators import BlockOperator, GradientOperator, IdentityOperator, FiniteDifferenceOperator

# Plugin to show gradient image with totalvariance regularisation applied
from cil.plugins.ccpi_regularisation.functions import FGP_TV


# Forward/backprojector from CIL astra plugin
from cil.plugins.astra import ProjectionOperator

# Import Matlab stopping rules
#import matlab.engine

# All external imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy.io
import os
from PIL import Image
from functools import partial
from types import MethodType
import pandas as pd

# remove some annoying warnings
import logging
logger = logging.getLogger('dxchange')
logger.setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING)
cil_log_level = logging.getLogger('cil.processors')
cil_log_level.setLevel(logging.INFO)

#%%read in the Data
scanner = 'tescan'

savedir = r'./'

waygatedata = ' path to waygate data'
tescandata = ' path to waygate data'
#%%
print('Data Read Start')
if scanner == 'waygate':
    path = waygatedata
    filename = os.path.join(path, r"RTX_160523_W3_cimat_Lego_EvK.pca")
    data = WaygateDataReader(file_name = filename).read()
elif scanner == 'tescan':
        path = tescandata
        filename = os.path.join(path, r"Acquisition settings XRE.txt")
        data = TescanDataReader(file_name = filename).read()
print('Data Read Finish')

type(data)
print(data)
print(data.geometry)
#show_geometry(data.geometry)

#%%Perfomr flat/dark field corrections FOR TESCAN
if scanner == 'tescan':
    print('Normalisation started')
    flat1 = plt.imread(os.path.join(path,"io000000.tif")).astype(np.float32)
    dark1 = plt.imread(os.path.join(path,"di000000.tif")).astype(np.float32)
    data = Normaliser(flat_field=flat1,
                       dark_field=dark1
                      )(data)
    
    print('Normalisation Finished')

#%%
# Display the projection data
print('Beer lambert start')
Binner({'horizontal':(None,None,2),'vertical':(None,None,2)})

print('Beer lambert start')
show2D(data, origin='upper-left')


# Convert to transmission imaages not absorption
data= TransmissionAbsorptionConverter(min_intensity=2**(-52))(data)
print('Beer lambert start')


show2D(data, origin='upper-left')
print('Beer lambert finish')
#%% Perform Reconstruction
print('geometry')


# # Reconstruction using FDK
data.reorder(order = 'tigre')

# Center of rotation correction(can either use xcorrelation, used fo parrallel beams, or image_sharpness, for cone beam)
data = CentreOfRotationCorrector.image_sharpness(backend='tigre', search_range=100, tolerance=0.1)(data)
#%%
ig = data.geometry.get_ImageGeometry()
fdk = FDK(data, ig)                                             
recon_centred = fdk.run()
#%%
show2D(recon_centred,
        origin='upper-right', num_cols=1 )

#%%

TIFFWriter(data=recon_centred, file_name=os.path.join(savedir, "recon "+ scanner, "FDK")).write()


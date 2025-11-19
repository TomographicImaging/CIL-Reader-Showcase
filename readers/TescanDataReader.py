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
# Evan Kiely (Warwick Manufacturing Group, University of Warwick)
# Ford Collins (Warwick Manufacturing Group, University of Warwick)
# Nishitha Ravichandran (Warwick Manufacturing Group, University of Warwick)
# Jay Warnett (Warwick Manufacturing Group, University of Warwick)
# Evelien Zwanenburg (Warwick Manufacturing Group, University of Warwick)

from cil.framework import AcquisitionData, AcquisitionGeometry
from readers.WMG_Modified_TIFF_io import TIFFStackReader
from cil.processors import  Normaliser
import warnings
import numpy as np
import os
from PIL import Image
import glob     
        
class TescanDataReader(object):
    
    def __init__(self, file_name = None, roi= None,
               normalise=True, mode='bin', fliplr=False):

      self.file_name = file_name
      self.roi = roi
      self.normalise = normalise
      self.mode = mode
      self.fliplr = fliplr

      if file_name is not None:
          self.set_up(file_name = file_name,
                      roi = roi,
                      normalise = normalise,
                      mode = mode,
                      fliplr = fliplr)
            
    def set_up(self, 
               file_name = None, 
               roi = {'angle': -1, 'horizontal': -1, 'vertical': -1},
               normalise = True,
               mode = 'bin',
               fliplr = False, 
               ):
            
        self.file_name = file_name
        self.roi = roi
        self.normalise = normalise
        self.mode = mode
        self.fliplr = fliplr
        self.tiff_directory_path = os.path.dirname(self.file_name)
        
        print(self.file_name)

        if self.file_name == None:
            raise Exception('Path to file is required.')
        
        # check if data file exists
        if not(os.path.isfile(self.file_name)):
            raise Exception('File\n {}\n does not exist.'.format(self.file_name))
        
        if self.roi is None:
           self.roi= {'angle': -1, 'horizontal': -1, 'vertical': -1}
           
        # check labels     
        for key in self.roi.keys():
            if key not in ['angle', 'horizontal', 'vertical']:
                raise Exception("Wrong label. One of the following is expected: angle, horizontal, vertical")
        
        roi = self.roi.copy()
        
        if 'angle' not in roi.keys():
            roi['angle'] = -1
            
        if 'horizontal' not in roi.keys():
            roi['horizontal'] = -1
        
        if 'vertical' not in roi.keys():
            roi['vertical'] = -1
                
        # parse data file
        with open(self.file_name, 'r', errors='replace') as f:
            content = f.readlines()    
                
        content = [x.strip() for x in content]
        

        for line in content:
            # number of projections
            if line.startswith("total projections "):
                line = line.replace('\"', ' ')
                num_projections = int(line.split('=')[1])
                num_projections = num_projections-1
            # number of pixels along Y axis
            elif line.startswith("Columns "):
                line = line.replace('\"', ' ')
                pixel_num_h_0 = int(line.split('=')[1])
            # number of pixels along X axis
            elif line.startswith("Rows "):
                line = line.replace('\"', ' ')
                pixel_num_v_0 = int(line.split('=')[1])
            # pixel size along X and y axis
            elif line.startswith("Binned pixelsize (mm) "):
                line = line.replace('\"', ' ')
                pixel_size_h_0 = float(line.split('=')[1])
                pixel_size_v_0 = pixel_size_h_0
            elif line.startswith("SOD "):
                line = line.replace('\"', ' ')
                source_to_origin = float(line.split('=')[1])
            # source to detector distance
            elif line.startswith("SDD "):
                line = line.replace('\"', ' ')
                source_to_det = float(line.split('=')[1])
            # initial angular position of a rotation stage
            elif line.startswith("CT start angle "):
                line = line.replace('\"', ' ')
                initial_angle = float(line.split('=')[1])
            # angular increment (in degrees)
            elif line.startswith("CT stop angle "):
                line = line.replace('\"', ' ')
                final_angle = float(line.split('=')[1])
            elif line.startswith("basename "):
                line = line.replace('\"', '')
                self.proj_name = (line.split('= ')[1])
                self.tiff_directory_path = os.path.join(os.path.dirname(self.file_name))
                  
        print(num_projections)
        
        if roi['angle'] == -1:
            roi['angle'] = (0,num_projections)
        self._roi_par = [[0, num_projections, 1] ,[0, pixel_num_v_0, 1], [0, pixel_num_h_0, 1]]
        
        for key in roi.keys():
            if key == 'angle':
                idx = 0
            elif key == 'vertical':
                idx = 1
            elif key == 'horizontal':
                idx = 2
            if roi[key] != -1:
                for i in range(2):
                    if roi[key][i] != None:
                        if roi[key][i] >= 0:
                            self._roi_par[idx][i] = roi[key][i]
                        else:
                            self._roi_par[idx][i] = self._roi_par[idx][1]+roi[key][i]
                if len(roi[key]) > 2:
                    if roi[key][2] != None:
                        if roi[key][2] > 0:
                            self._roi_par[idx][2] = roi[key][2] 
                        else:
                            raise Exception("Negative step is not allowed")
        
        if self.mode == 'bin':
            # calculate number of pixels and pixel size
            pixel_num_v = (self._roi_par[1][1] - self._roi_par[1][0]) // self._roi_par[1][2]
            pixel_num_h = (self._roi_par[2][1] - self._roi_par[2][0]) // self._roi_par[2][2]
            pixel_size_v = pixel_size_v_0 * self._roi_par[1][2]
            pixel_size_h = pixel_size_h_0 * self._roi_par[2][2]
        else: # slice
            pixel_num_v = int(np.ceil((self._roi_par[1][1] - self._roi_par[1][0]) / self._roi_par[1][2]))
            pixel_num_h = int(np.ceil((self._roi_par[2][1] - self._roi_par[2][0]) / self._roi_par[2][2]))

            pixel_size_v = pixel_size_v_0
            pixel_size_h = pixel_size_h_0
    

        angles = np.linspace(initial_angle,final_angle,num_projections,endpoint=False)
        
        if self.mode == 'bin':
            n_elem = (self._roi_par[0][1] - self._roi_par[0][0]) // self._roi_par[0][2]
            shape = (n_elem, self._roi_par[0][2])
            angles = angles[self._roi_par[0][0]:(self._roi_par[0][0] + n_elem * self._roi_par[0][2])].reshape(shape).mean(1)
        else:
            angles = angles[slice(self._roi_par[0][0], self._roi_par[0][1], self._roi_par[0][2])]
        


        if self.fliplr:
            origin = 'top-right'
        else:
            origin = 'top-left'
            

        if pixel_num_v == 1 and (self._roi_par[1][0]+self._roi_par[1][1]) // 2 == pixel_num_v_0 // 2:
            self._ag = AcquisitionGeometry.create_Cone2D(source_position=[0, -source_to_origin],
                                                     rotation_axis_position=[0, 0],
                                                     detector_position=[0, source_to_det-source_to_origin])
            self._ag.set_angles(angles, 
                                angle_unit='degree')
            
            self._ag.set_panel(pixel_num_h, pixel_size=pixel_size_h, origin=origin)

            self._ag.set_labels(labels=['angle', 'horizontal'])
        else:
            self._ag = AcquisitionGeometry.create_Cone3D(source_position=[0, -source_to_origin, 0],
                                                         rotation_axis_position=[0, 0, 0],
                                                         rotation_axis_direction=[0,0,1],
                                                         detector_position=[0, source_to_det-source_to_origin, 0])
            self._ag.set_angles(angles, 
                                angle_unit='degree')
            
            self._ag.set_panel((pixel_num_h, pixel_num_v),
                               pixel_size=(pixel_size_h, pixel_size_v),
                               origin=origin)
        
            self._ag.set_labels(labels=['angle', 'vertical', 'horizontal'])

    def get_geometry(self):
        
        '''
        Return AcquisitionGeometry object
        '''
        
        return self._ag
    def get_roi(self):
        '''returns the roi'''
        roi = self._roi_par[:]
        if self._ag.dimension == '2D':
            roi.pop(1)

        roidict = {}
        for i,el in enumerate(roi):
            # print (i, el)
            roidict['axis_{}'.format(i)] = tuple(el)
        return roidict
    
    def load_darkfield(self):
        di_path=glob.glob(os.path.join(self.tiff_directory_path,"di*"))[0]
        di = np.asarray(Image.open(di_path)).astype(np.float32)
        roi = self.get_roi()
        
        return di[roi['axis_1'][0]:roi['axis_1'][1],roi['axis_2'][0]:roi['axis_2'][1]]
        
    def load_flatfield(self):  
        io_path=glob.glob(os.path.join(self.tiff_directory_path,"io*"))[0]
        io = np.asarray(Image.open(io_path)).astype(np.float32)
        roi = self.get_roi()
       
        return io[roi['axis_1'][0]:roi['axis_1'][1],roi['axis_2'][0]:roi['axis_2'][1]]
    
    def read(self):
        
        '''
        Reads projections and return AcquisitionData container
        '''
        
        reader = TIFFStackReader()

        roi = self.get_roi()

        reader.set_up(file_name = self.tiff_directory_path,
                      roi=roi, mode=self.mode, proj_name = self.proj_name)

        ad = reader.read_as_AcquisitionData(self._ag)
              
        if (self.normalise):
            ad.array[ad.array < 1] = 1
            di = self.load_darkfield()
            io = self.load_flatfield()
            
            ad = Normaliser(flat_field=io,
                    dark_field=di
                    )(ad)
            
            
            # cast the data read to float32
        
        if self.fliplr:
            dim = ad.get_dimension_axis('horizontal')
            ad.array = np.flip(ad.array, dim)
        
        return ad

    def load_projections(self):
        '''alias of read for backward compatibility'''
        return self.read()



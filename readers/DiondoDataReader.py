#   Copyright 2025 UKRI-STFC
#   Copyright 2025 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# Authors:
# Evangelos Papoutsellis,  Finden Ltd 
# Kubra Kumrular, University of Southampton, μ-VIS X-ray Imaging Centre 

import os
import xml.etree.ElementTree as ET
import numpy as np
import glob
from cil.framework import AcquisitionGeometry, AcquisitionData
import threading
# first test it, create 2 options
from joblib import Parallel, delayed # parallel uploding 
import multiprocessing


class DiondoDataReader(object):
    
    """
     Fix Documentation
     Create a reader Diondo XML metadata and projection data (RAW format)

    Parameters
    ----------
    xml_file : str
        Path to the Diondo XML metadata file.
    projection_path : str
        Folder containing the corresponding RAW projection files.
    """

    def __init__(self, xml_file=None, projection_path=None):
        
        self.xml_file        = xml_file
        self.projection_path = projection_path

        if xml_file is not None:
            self.set_up()        

    def set_up(self):

        self.read_metadata()
        self._setup_geometry()
        
    def read_metadata(self):
        if not os.path.isfile(self.xml_file):
            raise FileNotFoundError("XML file not found.")

        tree = ET.parse(self.xml_file)
        root = tree.getroot()

        # Get information from Geometrie (Detector)
        geom = root.find("Geometrie")
        
        self.source_to_detector = float(geom.find("SourceDetectorDist").text)
        self.source_to_object   = float(geom.find("SourceObjectDist").text)
        self.object_to_detector = float(geom.find("ObjectDetectorDist").text)

        # Get information from Recon (Recon volume)
        recon = root.find("Recon")

        self.num_projections    = int(recon.find("ProjectionCount").text)
        self.num_pixels_h       = int(recon.find("ProjectionDimX").text)
        self.num_pixels_v       = int(recon.find("ProjectionDimY").text)        
        self.pixel_size_h       = float(recon.find("ProjectionPixelSizeX").text)
        self.pixel_size_v       = float(recon.find("ProjectionPixelSizeY").text)

    def _setup_geometry(self):
        """Creates the 3D AcquisitionGeometry object based on parsed data."""
        
        self.full_geometry = AcquisitionGeometry.create_Cone3D(
            source_position=[0, -self.source_to_object, 0],
            detector_position=[0, self.source_to_detector - self.source_to_object, 0]
        )
        self.full_geometry.set_panel((self.num_pixels_h, self.num_pixels_v),
                                pixel_size=(self.pixel_size_h, self.pixel_size_v), 
                                     origin='top-left')
        self.full_geometry.set_labels(['angle', 'vertical', 'horizontal'])

        # fix with initial angle???
        self.angles = np.linspace(-90, 270, self.num_projections, endpoint=False) 
        
        self.full_geometry.set_angles(self.angles, angle_unit='degree')                

    def get_metadata(self):
        pass
     
    def _read_single_raw_file(self, raw_file, height, width, dtype):
        """Read one single raw file"""
        with open(raw_file, "rb") as f:
            return np.fromfile(f, dtype=dtype).reshape((height, width)) 

    def read(self, angles=None, dtype=np.uint16, fread=False):
    
        raw_files = sorted(glob.glob(os.path.join(self.projection_path, "*.raw")))
        raw_files = raw_files[:-1]
    
        if len(raw_files) == 0:
            raise FileNotFoundError("No .raw files found in projection_path.")
    
        if angles is None:
            indices = np.arange(len(raw_files), dtype=int)
        else:
            # angles input by the user
            indices = self.set_angles(angles)
    
        selected_files = [raw_files[i] for i in indices]
        
        if fread:
            n_threads = max(1, multiprocessing.cpu_count() // 2)

            projection_data_lst_np = Parallel(n_jobs=n_threads, backend='threading')(delayed(self._read_single_raw_file)(
                    raw_file, self.num_pixels_v, self.num_pixels_h, dtype
                )
                for raw_file in selected_files
            )
            projection_data = np.stack(projection_data_lst_np, axis=0)
        else:
            projection_data = np.empty((len(selected_files), self.num_pixels_v, self.num_pixels_h), dtype=dtype)
            for k, raw_file in enumerate(selected_files):
                print(k, end="\r")
                projection_data[k] = self._read_single_raw_file(
                    raw_file, self.num_pixels_v, self.num_pixels_h, dtype
                )
    
        # Geometry subset: use the same indices
        geom = self.full_geometry.copy()
        geom.set_angles(self.full_geometry.angles[indices], angle_unit='degree')
    
        return AcquisitionData(array=projection_data.astype("float32"), geometry=geom)

    def set_angles(self, selection):
    
        if selection is None:
            self._indices = None
            return None

        if isinstance(selection, tuple):
            selection = slice(*selection)
        if isinstance(selection, slice):
            indices = np.arange(self.num_projections, dtype=int)[selection]
        elif isinstance(selection, (int, np.integer)):
            indices = np.array([int(selection)], dtype=int)
        else:
            angles = np.asarray(selection, dtype=float)
            indices_float = angles / 360.0 * self.num_projections     
            indices = np.round(indices_float).astype(int)

        # print(indices)
        # print(self.full_geometry.angles[indices])
    
        return indices
        

    def read_centre_slice(self, angles=None, dtype=np.uint16):
        """Reads only the central slice from each RAW projection."""

        geom_shape = self.full_geometry.shape
        central_index = geom_shape[1] // 2  

        self.raw_files = sorted(glob.glob(os.path.join(self.projection_path, "*.raw")))
        self.raw_files = self.raw_files[:-1]

        geometry2D = self.full_geometry.get_slice(vertical = "centre") 

        ### use central_slices and then create AcquisitionData --> should be faster vs .fill
        central_slices = np.zeros((geom_shape[0], geom_shape[2]), dtype=dtype)

        ### maybe add joblib/dask option
        for i, raw_file in enumerate(self.raw_files):
            with open(raw_file, "rb") as f:
                f.seek(central_index * geom_shape[2] * np.dtype(dtype).itemsize)
                central_slices[i] = np.fromfile(f, dtype=dtype, count=geom_shape[2])
        
        return AcquisitionData(array=np.float32(central_slices), geometry=geometry2D)

        ### Alternative
        # data2D = geometry2D.allocate()        

        # for i, raw_file in enumerate(self.raw_files):
        #     with open(raw_file, "rb") as f:
        #        f.seek(central_index * geom_shape[2] * np.dtype(dtype).itemsize)

        #     row_data = np.fromfile(raw_file, dtype=dtype, 
        #                            offset = central_index * geom_shape[2] * np.dtype(dtype).itemsize,
        #                            count=geom_shape[2])
        #     data2D.fill(row_data, angle=i)
        # return data2D
        
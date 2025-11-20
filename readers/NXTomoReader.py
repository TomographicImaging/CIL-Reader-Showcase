#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

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
# Laura Murgatroyd (URKI-STFC)
# Edoardo Pasca (UKRI-STFC)
# Srikanth Nagella (UKRI-STFC)
# Gemma Fardell (UKRI-STFC)
# Evelina Ametova (The University of Manchester)
# Mike Sullivan (UKRI-STFC)
# Sam Tygier (UKRI-STFC)

import math
import os
import h5py
from cil.framework import AcquisitionGeometry
import numpy as np

h5pyAvailable = True
try:
    from h5py import File as NexusFile
except ImportError:
    h5pyAvailable = False


TOMO_ENTRY = "tomo_entry"
DATA_PATH = "instrument/detector/data"
IMAGE_KEY_PATH = "instrument/detector/image_key"
ROTATION_ANGLE_PATH = "sample/rotation_angle"
DEFINITION = "definition"
NXTOMOPROC = "NXtomoproc"


class NXTomoReader(object):
    '''
    Reader class for loading Nexus files in the NXTomo format:
    https://manual.nexusformat.org/classes/applications/NXtomo.html
    '''
    
    def __init__(self, file_name=None, normalise = True):
        '''
        This takes in input as file_name and loads the dataset.
        '''

        self.setup(file_name=file_name,
               normalise = normalise)


    def setup(self,
               file_name = None,
               roi = None,
               normalise = True,
               mode = 'bin'): # CHECK INPUTS ARE VALID
        
        self.file_name = file_name
        self.roi = roi
        self.normalise = normalise
        self.mode = mode
        self.flat = None
        self.dark = None
        self.angles = None
        self.geometry = None

        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        
        if self.file_name is None:
            raise ValueError('Path to NXTomo file is required.')

        # check if file exists
        if not(os.path.isfile(self.file_name)):
            raise FileNotFoundError('File\n {}\n does not exist.'.format(self.file_name))

        self.tomo_path = self._get_nxtomo_entry_path()
        assert self.tomo_path is not None, "No tomo_entry found in file."

        self.data_path = f"{self.tomo_path}/{DATA_PATH}"
        self.key_path = f"{self.tomo_path}/{IMAGE_KEY_PATH}"
        self.angle_path = f"{self.tomo_path}/{ROTATION_ANGLE_PATH}"  

        self._check_tomo_data_exists(self.data_path)
        self._check_tomo_data_exists(self.key_path)
        self._check_tomo_data_exists(self.angle_path)

        self.rotation_angles, angle_unit = self.get_rotation_angles()

        self.full_dims = self._get_projection_dimensions()

        try:
            x_pixel_size = self._get_tomo_data_as_array(f"{self.tomo_path}/instrument/detector/x_pixel_size")
            x_pixel_size = float(x_pixel_size[0])
        except (KeyError, ValueError):
            x_pixel_size = 1
        
        try:
            y_pixel_size = self._get_tomo_data_as_array(f"{self.tomo_path}/instrument/detector/y_pixel_size")
            y_pixel_size = float(y_pixel_size[0])
        except (KeyError, ValueError):
            y_pixel_size = 1
        
        geometry = AcquisitionGeometry.create_Parallel3D().set_panel(
        num_pixels=(self.full_dims[2], self.full_dims[1]), pixel_size=(x_pixel_size, y_pixel_size)).set_angles(
        angles=self.get_projection_angles()).set_labels(
        ['angle', 'vertical', 'horizontal']).set_channels(1)

        self.geometry = geometry

    def get_image_keys(self):
        '''
        Loads the image keys from the nexus file
        returns: numpy array containing image keys:
        0 = projection
        1 = flat field
        2 = dark field
        3 = invalid
        '''
        
        with NexusFile(self.file_name, "r") as nexus_file:
            return nexus_file[self.key_path][:]

    def load(self, dimensions=None, image_key_id=0):
        '''
        This is generic loading function of flat field, dark field and
        projection data.
        Loads data with image key id of image_key_id
        dimensions: a tuple of 'slice'
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        '''
        try:
            with NexusFile(self.file_name, 'r') as file:
                image_keys = self.get_image_keys()
                if dimensions is None:
                    projections = np.array(file[self.data_path])
                    result = projections[image_keys == image_key_id]
                    return result
                else:
                    # First dimension shifted to take into account image_key (e.g. projections may not start at zero)
                    index_array = np.where(image_keys == image_key_id)
                    projection_indexes = index_array[0][dimensions[0]]
                    new_dimensions = list(dimensions)
                    new_dimensions[0] = projection_indexes
                    new_dimensions = tuple(new_dimensions)
                    result = np.array(file[self.data_path][new_dimensions])
                    return result
        except Exception:
            print("Error reading nexus file")
            raise

    def _get_nxtomo_entry_path(self) -> h5py.Group | None:
        """
        Look for a tomo_entry field in the NeXus file. Generate an error if it can't be
        found.
        :return: The first tomo_entry group if one could be found, None otherwise.
        """
        with NexusFile(self.file_name, "r") as self.nexus_file:
            for key in self.nexus_file.keys():
                if TOMO_ENTRY in self.nexus_file[key].keys():
                    entry = self.nexus_file[key][TOMO_ENTRY]
                    if isinstance(entry, h5py.Group):
                        tomo_path = f"{key}/{TOMO_ENTRY}"
                        return tomo_path
            return None

    def _get_tomo_data_as_array(self, entry_path: str) -> h5py.Group | h5py.Dataset | None:
        """
        Retrieve data from the tomo entry field.
        :param entry_path: The path in which the data is found.
        :return: The Nexus Group/Dataset if it exists, None otherwise.
        """
        with NexusFile(self.file_name, "r") as nexus_file:
            assert nexus_file[entry_path] is not None
            return nexus_file[entry_path][:].copy()

    def get_rotation_angles(self) -> np.ndarray:
        from cil.framework.labels import AngleUnit
        with NexusFile(self.file_name, "r") as nexus_file:

            rotation_angles = nexus_file[self.angle_path]

            if "units" not in rotation_angles.attrs.keys():
                print("No unit information found for rotation angles. Will infer from array values.")
                degrees = np.abs(rotation_angles).max() > 2 * np.pi            
            else:
                degrees = "deg" in str(rotation_angles.attrs["units"])

            if degrees:
                units = AngleUnit.DEGREE
            else:
                units = AngleUnit.RADIAN

            return rotation_angles[:].copy(), units
           
        
    def _check_tomo_data_exists(self, entry_path: str) -> bool:
        """
        Check if data exists in the tomo entry field.
        :param entry_path: The path in which the data is found.
        :return: True if the data exists, False otherwise.
        """
        with NexusFile(self.file_name, "r") as nexus_file:
            if isinstance(nexus_file[entry_path], h5py.Dataset | h5py.Group):
                return True
            else:
                raise KeyError(f"{entry_path} does not exist in {self.file_name}")

    def load_projection(self, dimensions=None):
        '''
        Loads the projection data from the nexus file.
        dimensions: a tuple of 'slice's
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        returns: numpy array with projection data
        '''
        try:
            if 0 not in self.get_image_keys():
                raise ValueError("Projections are not in the data. Data Path ",
                                 self.data_path)
        except KeyError as ke:
            raise KeyError(ke.args[0], self.data_path)
        return self.load(dimensions, 0)

    def load_flat_field(self, dimensions=None): # TODO: convert to use CIL's ROI
        '''
        Loads the flat field data from the nexus file.
        dimensions: a tuple of 'slice's
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        returns: numpy array with flat field data
        '''
        try:
            if 1 not in self.get_image_keys():
                raise ValueError("Flats are not in the data. Data Path ",
                                 self.data_path)
        except KeyError as ke:
            raise KeyError(ke.args[0], self.data_path)
        return self.load(dimensions, 1)

    def load_dark_field(self, dimensions=None): # TODO: convert to use CIL's ROI
        '''
        Loads the Dark field data from the nexus file.
        dimensions: a tuple of 'slice's
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        returns: numpy array with dark field data
        '''
        try:
            if 2 not in self.get_image_keys():
                raise ValueError("Darks are not in the data. Data Path ",
                                 self.data_path)
        except KeyError as ke:
            raise KeyError(ke.args[0], self.data_path)
        return self.load(dimensions, 2)

    def get_projection_angles(self):
        '''
        Loads the projection angles from the nexus file.
        returns: array containing the projection angles
        '''
        if self.file_name is None:
            return
        try:
            with NexusFile(self.file_name, 'r') as file:
                angles = np.array(file[self.angle_path], np.float32)
                image_keys = np.array(file[self.key_path])
                return angles[image_keys == 0]
        except Exception:
            print("get_projection_angles Error reading nexus file")
            raise

    def get_sinogram_dimensions(self):
        '''
        Return the sinogram dimensions of the dataset
        '''
        if self.file_name is None:
            return
        try:
            with NexusFile(self.file_name, 'r') as file:
                projections = file[self.data_path]
                image_keys = np.array(file[self.key_path])
                dims = list(projections.shape)
                dims[0] = dims[1]
                dims[1] = np.sum(image_keys == 0)
                return tuple(dims)
        except Exception:
            print("Error reading nexus file")
            raise

    def _get_projection_dimensions(self):
        '''
        Return the projection dimensions of the dataset
        '''
        with NexusFile(self.file_name, 'r') as file:
            projections = file[self.data_path]
            image_keys = self.get_image_keys()
            dims = list(projections.shape)
            dims[0] = np.sum(image_keys == 0)
            return tuple(dims)

    def get_geometry(self, dimensions=None):
        if dimensions is not None:
            dim_h = (dimensions[2].stop - dimensions[2].start)
            if dimensions[2].step is not None:
                dim_h = math.ceil(dim_h / dimensions[2].step)
            dim_v = (dimensions[1].stop - dimensions[1].start)
            if dimensions[1].step is not None:
                dim_v = math.ceil(dim_v / dimensions[1].step)
            geometry = AcquisitionGeometry.create_Parallel3D().set_panel(
                num_pixels=(dim_h, dim_v), pixel_size=(1.0, 1.0)).set_angles(
                angles=self.get_projection_angles()[dimensions[0]]).set_labels(
                ['angle', 'vertical', 'horizontal']).set_channels(1)
            return geometry

        return self.geometry

    def read(self, dimensions=None):
        '''
        Loads the acquisition data within the given dimensions and returns
        an AcquisitionData Object
        dimensions: a tuple of 'slice'
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        '''
        data = self.load_projection(dimensions)
        geometry = self.get_geometry(dimensions)
        out = geometry.allocate()
        out.fill(data)
        return out

    def get_acquisition_data_subset(self, ymin=None, ymax=None):
        '''
        This method load the acquisition data, cropped in the vertical
        direction, from ymin to ymax, and returns
        an AcquisitionData object
        '''
        return self.read(dimensions=(slice(0, self.full_dims[0]), slice(ymin, ymax), slice(0, self.full_dims[2])))

    def get_acquisition_data_slice(self, y_slice=0):
        ''' Returns a vertical slice of the projection data at y_slice,
         as an AcquisitionData object'''
        return self.get_acquisition_data_subset(ymin=y_slice, ymax=y_slice+1)

    def list_file_content(self):
        '''
        Prints paths to all datasets within nexus file.
        '''
        try:
            with NexusFile(self.file_name, 'r') as file:
                file.visit(print)
        except Exception:
            print("Error reading nexus file")
            raise

    def get_acquisition_data_batch(self, bmin=None, bmax=None):
        # TODO: Perhaps we should rename?
        '''
        This method load the acquisition data, cropped in the angular
        direction, from index bmin to bmax, and returns
        an AcquisitionData object
        '''
        return self.read(dimensions=(slice(bmin, bmax), slice(0, self.full_dims[1]), slice(0, self.full_dims[2])))


if __name__ == "__main__":
    file_path = 'data/24737_fd.nxs'

    reader = NXTomoReader(file_name=file_path)
    acq_data = reader.read()
    print(f"{acq_data.shape=}")
    flat = reader.load_dark_field()
    from cil.utilities.display import show2D

    acq_data_sub = reader.read(dimensions=(slice(0,10), slice(0,135,1), slice(0,160,3)))
    print(f"{acq_data_sub.shape=}")

    acq_data_sub_y = reader.get_acquisition_data_subset(50, 100)
    print(f"{acq_data_sub_y.shape}")

    acq_data_sub_angle = reader.get_acquisition_data_batch(50, 60)
    print(f"{acq_data_sub_angle.shape}")

    show2D(acq_data_sub_angle)

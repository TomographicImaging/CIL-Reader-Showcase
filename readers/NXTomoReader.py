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
from cil.framework import AcquisitionGeometry
import numpy as np
from cil.io.utilities import HDF5_utilities

h5pyAvailable = True
try:
    import h5py
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

    Parameters
    ----------
    file_name : str
        Path to the Nexus file to be read.
    '''
    
    def __init__(self, file_name=None):

        self.setup(file_name=file_name)


    def setup(self, file_name = None):
        '''
        Parameters
        ----------
        file_name : str
            Path to the Nexus file to be read.
        '''        
        self.file_name = file_name
        self.flat = None
        self.dark = None
        self.angles = None
        self.geometry = None

        if not h5pyAvailable:
            raise ImportError("Error: h5py is not installed")
        
        if self.file_name is None:
            raise ValueError('Path to NXTomo file is required.')

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

        rotation_angles, angle_unit = self._get_rotation_angles_and_unit()

        full_dims = self._get_projection_dimensions()

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
        num_pixels=(full_dims[2], full_dims[1]), pixel_size=(x_pixel_size, y_pixel_size)).set_angles(
        angles=rotation_angles, angle_unit=angle_unit).set_labels(
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

        Parameters
        ----------
        dimensions : tuple of slice, optional
            e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
            The dimensions to load from the data. The default is None, which loads all data with the specified image key.
        image_key_id : int, optional
            The image key id to load. The default is 0, which corresponds to projection data.
        
        Returns
        -------
        numpy array with data corresponding to the specified image key and dimensions.
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

        Returns
        -------
        The first tomo_entry group if one could be found, None otherwise.
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
        Parameters
        ----------
        entry_path:
            The path in which the data is found.
        Returns
        -------
        The data as a numpy array.
        """
        with NexusFile(self.file_name, "r") as nexus_file:
            assert nexus_file[entry_path] is not None
            return nexus_file[entry_path][:].copy()
        

        
    def _get_rotation_angles_and_unit(self):
        '''
        Returns
        -------
        A tuple containing:
        - numpy array with rotation angles
        - unit of the rotation angles (AngleUnit.DEGREE or AngleUnit.RADIAN)
        '''
        from cil.framework.labels import AngleUnit
        with NexusFile(self.file_name, "r") as nexus_file:

            rotation_angles = nexus_file[self.angle_path]

            if "units" not in rotation_angles.attrs.keys():
                # No unit information found for rotation angles. Will infer from array values
                degrees = np.abs(rotation_angles).max() > 2 * np.pi            
            else:
                degrees = "deg" in str(rotation_angles.attrs["units"])

            if degrees:
                units = AngleUnit.DEGREE
            else:
                units = AngleUnit.RADIAN

            image_keys = np.array(nexus_file[self.key_path])
            rotation_angles = rotation_angles[image_keys == 0]

            return rotation_angles.copy(), units
               
        
    def _check_tomo_data_exists(self, entry_path: str) -> bool:
        """
        Check if data exists in the tomo entry field.
        Parameters
        ----------
        enntry_path:
            The path to the data
        
        Returns
        -------
        True if the data exists, False otherwise.
        """
        with NexusFile(self.file_name, "r") as nexus_file:
            if isinstance(nexus_file[entry_path], h5py.Dataset | h5py.Group):
                return True
            else:
                raise KeyError(f"{entry_path} does not exist in {self.file_name}")


    def load_projection(self, dimensions=None):
        '''
        Loads the projection data from the nexus file.

        Parameters
        ----------
        dimensions : tuple of slice, optional
            e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
            The dimensions to load from the data. The default is None, which loads all projection data.

        Returns
        -------
        numpy array with projection data
        '''
        try:
            if 0 not in self.get_image_keys():
                raise ValueError("Projections are not in the data. Data Path ",
                                 self.data_path)
        except KeyError as ke:
            raise KeyError(ke.args[0], self.data_path)
        return self.load(dimensions, 0)

    def load_flat_field(self, dimensions=None):
        '''
        Loads the flat field data from the nexus file.

        Parameters
        ----------
        dimensions : tuple of slice, optional
            e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
            The dimensions to load from the data. The default is None, which loads all flat field data.

        Returns
        -------
        numpy array with flat field data
        '''
        try:
            if 1 not in self.get_image_keys():
                raise ValueError("Flats are not in the data. Data Path ",
                                 self.data_path)
        except KeyError as ke:
            raise KeyError(ke.args[0], self.data_path)
        return self.load(dimensions, 1)

    def load_dark_field(self, dimensions=None):
        '''
        Loads the dark field data from the nexus file.

        Parameters
        ----------
        dimensions : tuple of slice, optional
            e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
            The dimensions to load from the data. The default is None, which loads all dark field data.

        Returns
        -------
        numpy array with dark field data
        '''
        try:
            if 2 not in self.get_image_keys():
                raise ValueError("Darks are not in the data. Data Path ",
                                 self.data_path)
        except KeyError as ke:
            raise KeyError(ke.args[0], self.data_path)
        return self.load(dimensions, 2)
    
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
        '''
        Parameters
        ----------
        dimensions : tuple of slice, optional
            e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
            The dimensions to get the geometry for. The default is None, which returns the full geometry.

        Returns
        -------
        AcquisitionGeometry object
        '''
        if dimensions is not None:
            dim_h = (dimensions[2].stop - dimensions[2].start)
            if dimensions[2].step is not None:
                dim_h = math.ceil(dim_h / dimensions[2].step)
            dim_v = (dimensions[1].stop - dimensions[1].start)
            if dimensions[1].step is not None:
                dim_v = math.ceil(dim_v / dimensions[1].step)
            angles, units = self._get_rotation_angles_and_unit()
            geometry = AcquisitionGeometry.create_Parallel3D().set_panel(
                num_pixels=(dim_h, dim_v), pixel_size=(1.0, 1.0)).set_angles(
                angles=angles[dimensions[0]], angle_unit=units).set_labels(
                ['angle', 'vertical', 'horizontal']).set_channels(1)
            return geometry

        return self.geometry

    def read(self, dimensions=None):
        '''
        Loads the acquisition data within the given dimensions and returns
        an AcquisitionData Object

        Parameters
        ----------
        dimensions: a tuple of 'slice'
            e.g. (slice(0, 1), slice(0, 135), slice(0, 160))

        Returns
        -------
        AcquisitionData
        '''
        data = self.load_projection(dimensions)
        geometry = self.get_geometry(dimensions)
        out = geometry.allocate()
        out.fill(data)
        return out

    def print_metadata(self, group='/', depth=-1):
        """
        Prints the file metadata

        Parameters
        ----------
        filename: str
            The full path to the HDF5 file
        group: (str), default: '/'
            a specific group to print the metadata for, this defaults to the root group
        depth: int, default -1
            depth of group to output the metadata for, -1 is fully recursive
        """

        HDF5_utilities.print_metadata(self.file_name, group, depth)

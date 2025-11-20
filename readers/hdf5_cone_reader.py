#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
#  Copyright 2025 Technical University of Denmark
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
#   Authored by:    Hannah Robarts (UKRI-STFC)
#                   Laura Murgatroyd (UKRI-STFC)
#                   Margaret Duff (UKRI-STFC)

from cil.framework import AcquisitionData, AcquisitionGeometry, DataContainer
from cil.io.utilities import HDF5_utilities
import os 
from cil.processors import  Binner
import numpy as np
from cil.utilities.display import show2D
from copy import deepcopy
from cil.io import TIFFStackReader
import weakref
h5pyAvailable = True
try:
    import h5py
except:
    h5pyAvailable = False

class HDF5_ConeDataReader(object): 
    """
    HDF5 generic cone data reader

    """

    DISTANCE_UNIT_LIST = ['m','cm','mm','um']
    DISTANCE_UNIT_MULTIPLIERS = [1.0, 1e-2, 1e-3, 1e-6]
    ANGLE_UNIT_LIST = ['degree', 'radian']


    def __init__(self, file_name, dataset_path,
                 dimension_labels=['angle', 'vertical', 'horizontal'], 
                 distance_units= 'cm', angle_units = 'degree'):

        '''
        Parameters
        ----------
        file_name: string
            file name to read

        dataset_path: string
            Path to the datasets within the HDF5 file

        dimension_labels: tuple (optional)
            Labels describing the order in which the data is stored, 
            default is ('angle', 'vertical', 'horizontal')
        
        roi: dict, default None
            dictionary with roi to load for each axis:
            ``{'axis_labels_1': (start, end, step), 'axis_labels_2': (start, 
            end, step)}``. ``axis_labels`` are defined by AcquisitionGeometry 
            dimension labels.
        
        distance_units: string, default = 'cm'
            Specify the distance units to use for the geometry, must be one of 
            'm', 'cm','mm' or 'um'

        angle_units: string, default = 'degree'
            Specify the distance units to use for the geometry, must be one of 
            'degree' or 'radian'

        '''
        if h5pyAvailable is False:
            raise ImportError("h5py package is required to use HDF5 readers")

        self._dimension_labels = dimension_labels
        self._data_handle = self.data_handler(self._read_data)
        self.file_name = file_name
        self.reset()
        self._dataset_path = dataset_path
        self.flatfield_path = None
        self.darkfield_path = None
        self.channels=False

        self._metadata = {
            'pixel_size_x' : 1,
            'pixel_size_y' : 1,
            'sample_detector_distance' : 0,
            }
        if distance_units in self.DISTANCE_UNIT_LIST:
            self._metadata['distance_units'] = distance_units
            self.distance_unit_multiplier = 1/(self.DISTANCE_UNIT_MULTIPLIERS[
                self.DISTANCE_UNIT_LIST.index(distance_units)])
        else:
            raise ValueError("Distance units not recognised expected one \
                                 of {}, got {}".format(str(self.DISTANCE_UNIT_LIST), 
                                                       str(distance_units)))
        if angle_units in self.ANGLE_UNIT_LIST:
            self._metadata['angle_units'] = angle_units
        else:
            raise ValueError("Distance units not recognised expected one \
                                 of {}, got {}".format(str(self.ANGLE_UNIT_LIST), 
                                                       str(angle_units)))
    @property
    def number_of_datasets(self):
        return self._number_of_datasets
    
    class data_handler(object):
        """
        This class controls the reading, casting, normalising and caching of the dataset.
        It should not need modification as long as it is configured with 'read_data_method'.
        """

        def __init__(self, read_data_method):
            self._read_data = read_data_method
            self._array = None
            self.dtype = None
            self.roi = None


        @property
        def array(self):
            if self._array is None:
                return None
            else:
                return self._array()
        
        @array.setter
        def array(self, val):
            self._array = weakref.ref(val)


        def get_data(self, dtype=np.float32, roi=None):
            """
            Caches the previous read when possible
            """
            # cached must be same roi castable dtype
            if self.array is not None and self.roi==roi and np.can_cast(self.dtype,dtype,casting='safe'):

                self.array.astype(dtype, casting='safe',copy=False)
                self.dtype = self.array.dtype

                array = self.array

            else:
                array = self._read_data(dtype, roi)
                array = np.asarray(array, dtype=dtype)
                self.array = array
                self.dtype = array.dtype
                self.roi = roi

            return array


    @property
    def file_name(self):
        return self._file_name
    

    @file_name.setter
    def file_name(self, val):
        file_name_abs = os.path.abspath(val)
        
        if not(os.path.isfile(file_name_abs)):
            raise FileNotFoundError('{}'.format(file_name_abs))
        
        file_extension = os.path.basename(file_name_abs).split('.')[-1].lower()
        if file_extension not in self._supported_extensions:
            raise TypeError('This reader can only process files with extensions: {0}. Got {1}'.format(self._supported_extensions, file_extension))

        self._file_name = val

        #if filename changes then reset the file dependent members
        self._acquisition_geometry = False
        self._metadata = False
        self._data_handle._array = None


    @property
    def full_geometry(self):
        if not self._acquisition_geometry:
            self._create_full_geometry()
        return self._acquisition_geometry


    @property
    def metadata(self):
        if not self._metadata:
            self._read_metadata()
        return self._metadata
    
    def configure_channels(self, num_channels):
        '''
        Parameters
        ----------
        no_channels: int
            Number of channels in the data
        '''
        self._metadata['num_channels'] = num_channels 
        self.channels = True 

    
    def configure_pixel_sizes(self, pixel_size_x_path=None, pixel_size_y_path=None, 
                         pixel_size_x=None, pixel_size_y=None,
                         HDF5_units=None):
        '''
        Parameters
        ----------
        pixel_size_x_path: string
            Path to the x pixel size within the HDF5 file

        pixel_size_y_path: string
            Path to the y pixel size within the HDF5 file
            
        pixel_size_x: float
            Alternatively provide the x pixel size as a float
            
        pixel_size_y: float
            Alternatively provide the y pixel size as a float
        
        HDF5_units: string (optional)
            The pixel size distance units in the HDF5 file, must be one of 'm',
            'cm','mm' or 'um', if not specified the units will be read from 
            the dataset attribute
        '''
        if pixel_size_x_path is not None:

            if HDF5_units is None:
                with h5py.File(self._file_name, 'r') as f:
                    dset = f.get(pixel_size_x_path)
                    HDF5_units = dset.attrs['units']
        
            if HDF5_units in self.DISTANCE_UNIT_LIST:
                multiplier = self.distance_unit_multiplier* \
                self.DISTANCE_UNIT_MULTIPLIERS[self.DISTANCE_UNIT_LIST.index(HDF5_units)]
            else:
                raise ValueError("Distance units not recognised expected one \
                                 of {}, got {}".format(str(self.DISTANCE_UNIT_LIST), 
                                                       str(HDF5_units)))
                                
            self._metadata['pixel_size_x'] = multiplier*HDF5_utilities.read(self._file_name, 
                                                               pixel_size_x_path)
        elif pixel_size_x is not None:
            self._metadata['pixel_size_x'] = pixel_size_x # TODO: is a multiplier needed here?
        
        if pixel_size_y_path is not None:

            if HDF5_units is None:
                with h5py.File(self._file_name, 'r') as f:
                    dset = f.get(pixel_size_y_path)
                    HDF5_units = dset.attrs['units']
        
            if HDF5_units in self.DISTANCE_UNIT_LIST:
                multiplier = self.distance_unit_multiplier*\
                self.DISTANCE_UNIT_MULTIPLIERS[self.DISTANCE_UNIT_LIST.index(HDF5_units)]
            else:
                raise ValueError("Distance units not recognised expected one \
                                 of {}, got {}".format(str(self.DISTANCE_UNIT_LIST), 
                                                       str(HDF5_units)))
                                
            self._metadata['pixel_size_y'] = multiplier*HDF5_utilities.read(self._file_name, 
                                                               pixel_size_y_path)
        elif pixel_size_y is not None:
            self._metadata['pixel_size_y'] = pixel_size_y # TODO: is a multiplier needed here?
            

    def configure_angles(self, angles_path=None, angles=None, HDF5_units=None):
        '''
        Parameters
        ----------
        angles_path: string
            Path to the angles within the HDF5 file

        angles: ndarray
            Alternatively provide the angles as an array
        
        HDF5_units: string (optional)
            The angle units in the HDF5 file, must be one of 'degree' or 'radian', 
            if not specified the units will be read from the dataset
            attribute
        '''
        
        if isinstance(angles_path,(tuple,list)):
            angles_cat = []
            
            for x in angles_path:
                if angles is None:
                    a = HDF5_utilities.read(self._file_name, x)
                else:
                    a = angles

                if HDF5_units is None:
                    with h5py.File(self._file_name, 'r') as f:
                        dset = f.get(x)
                        HDF5_units = dset.attrs['units']
                
                if HDF5_units in ['degree', 'deg', 'radian', 'rad']:
                    if ((self._metadata['angle_units']=='degree') or (self._metadata['angle_units']=='deg')) and (HDF5_units=='radian'):
                        a = np.rad2deg(a)
                    elif ((self._metadata['angle_units']=='radian') or (self._metadata['angle_units']=='rad')) and (HDF5_units=='degree'):
                        a = np.deg2rad(a)
                    self._metadata['angles'] = a
                else:
                    raise ValueError("Angle units not recognised, expected one of \
                                        'degree' or 'radian', got {}".format(str(HDF5_units)))
                angles_cat = np.concatenate((angles_cat,a))
                self._metadata['angles'] = angles_cat

            
        else:
            if angles is None:
                a = HDF5_utilities.read(self._file_name, angles_path)
            else:
                a = angles

            if HDF5_units is None:
                with h5py.File(self._file_name, 'r') as f:
                    dset = f.get(angles_path)
                    HDF5_units = dset.attrs['units']
            
            if HDF5_units in ['degree', 'deg', 'radian', 'rad']:
                if ((self._metadata['angle_units']=='degree') or (self._metadata['angle_units']=='deg')) and (HDF5_units=='radian'):
                    a = np.rad2deg(a)
                elif ((self._metadata['angle_units']=='radian') or (self._metadata['angle_units']=='rad')) and (HDF5_units=='degree'):
                    a = np.deg2rad(a)
                self._metadata['angles'] = a
            else:
                raise ValueError("Angle units not recognised, expected one of \
                                    'degree' or 'radian', got {}".format(str(HDF5_units)))
                
    def configure_source_detector_distance(self, source_detector_distance_path=None,
                                           source_detector_distance=None,
                                       HDF5_units=None):
        '''
        Parameters
        ----------
        source_detector_distance_path: string
            Path to the source to detector distance value within the HDF5 file
        source_detector_distance: float
            Alternatively provide the source to detector distance as a float
        HDF5_units: string (optional)
            The angle units in the HDF5 file, must be one of 'm', 'cm', 'mm'
            or 'um', if not specified the units will be read from the dataset
            attribute
            
        '''         
        if source_detector_distance_path is not None:
            if HDF5_units is None:
                with h5py.File(self._file_name, 'r') as f:
                    dset = f.get(source_detector_distance_path)
                    HDF5_units = dset.attrs['units']
        
        if HDF5_units in self.DISTANCE_UNIT_LIST:
            multiplier = self.distance_unit_multiplier*\
            self.DISTANCE_UNIT_MULTIPLIERS[self.DISTANCE_UNIT_LIST.index(HDF5_units)]
        else:
            raise ValueError("Distance units not recognised expected one \
                            of {}, got {}".format(str(self.DISTANCE_UNIT_LIST), 
                                                str(HDF5_units)))
        if source_detector_distance is None:
            if source_detector_distance_path is None:
                raise("Please enter source_detector_distance or path")
            else:
                source_detector_distance = HDF5_utilities\
                .read(self._file_name, source_detector_distance_path)
        
        self._metadata['source_detector_distance'] = multiplier\
            *source_detector_distance
                
    def configure_sample_detector_distance(self, sample_detector_distance_path=None,
                                           sample_detector_distance=None,
                                       HDF5_units=None):
        '''
        Parameters
        ----------
        sample_detector_distance_path: string
            Path to the sample to detector distance value within the HDF5 file
        
        sample_detector_distance: float
            Alternatively provide the sample to detector distance as a float
            
        HDF5_units: string (optional)
            The angle units in the HDF5 file, must be one of 'm', 'cm', 'mm' 
            or 'um', if not specified the units will be read from the dataset
            attribute
        '''
        
        if sample_detector_distance_path is not None:
            if HDF5_units is None:
                with h5py.File(self._file_name, 'r') as f:
                    dset = f.get(sample_detector_distance_path)
                    HDF5_units = dset.attrs['units']
        
        if HDF5_units in self.DISTANCE_UNIT_LIST:
            multiplier = self.distance_unit_multiplier*\
            self.DISTANCE_UNIT_MULTIPLIERS[self.DISTANCE_UNIT_LIST.index(HDF5_units)]
        else:
            raise ValueError("Distance units not recognised expected one \
                            of {}, got {}".format(str(self.DISTANCE_UNIT_LIST), 
                                                str(HDF5_units)))
        if sample_detector_distance is None:
            if sample_detector_distance_path is None:
                raise("Please enter sample_detector_distance or path")
            else:
                sample_detector_distance = HDF5_utilities\
                .read(self._file_name, sample_detector_distance_path)
        
        self._metadata['sample_detector_distance'] = multiplier\
            *sample_detector_distance
        
    def configure_normalisation_data(self, filename=None,darkfield_path=None, 
                                     flatfield_path=None):
        if filename is None:
            self.norm_filename = self.file_name
        else:
            self.norm_filename = filename
        
        if flatfield_path is not None:
            self.flatfield_path = flatfield_path
            
        if darkfield_path is not None:
            self.darkfield_path = darkfield_path
            
    @property
    def _supported_extensions(self):
        """A list of file extensions supported by this reader"""
        return ['hdf5','h5']
    
    def _read_metadata(self):
        """
        Gets the `self._metadata` dictionary of values from the dataset meta 
        data. The metadata is created using specific configure methods like 
        `configure_pixel_sizes` which take the path to the meta data within
        the HDF5 file.
        """
        self._metadata = self._metadata

    def _get_shape(self):
        ds_metadata = HDF5_utilities.get_dataset_metadata(self._file_name, 
                                                          self._dataset_path)
        vertical= ds_metadata['shape'][self._dimension_labels.index('vertical')]
        #What is this for? Is it used anywhere? Probably not finished 

    def _create_full_geometry(self):
        """
        Create the `AcquisitionGeometry` `self._acquisition_geometry` that 
        describes the full dataset.

        This should use the values from `self._metadata` where possible.
        """
        if isinstance(self._dataset_path,(tuple,list)):

            for i, x in enumerate(self._dataset_path):
                if i == 0:
                    ds_metadata = HDF5_utilities.get_dataset_metadata(self._file_name, 
                                                          x)
                # else:
                #     ds_metadata_test = HDF5_utilities.get_dataset_metadata(self._file_name, 
                #                                             x)

                #     if (ds_metadata['shape'][self._dimension_labels.index('vertical')])!=(ds_metadata_test['shape'][self._dimension_labels.index('vertical')]):
                #         raise ValueError('Datasets must the same shape')
                #     if len(ds_metadata['shape']) > 2:
                #         if (ds_metadata['shape'][self._dimension_labels.index('horizontal')])!=(ds_metadata_test['shape'][self._dimension_labels.index('horizontal')]):
                #             raise ValueError('Datasets must the same shape.')
        else:
            ds_metadata = HDF5_utilities.get_dataset_metadata(self._file_name, 
                                                          self._dataset_path)
    
        horizontal= ds_metadata['shape'][self._dimension_labels.index('horizontal')]

        if len(ds_metadata['shape']) > 2+self.channels:
            vertical = ds_metadata['shape'][self._dimension_labels.index('vertical')]

            self._acquisition_geometry = AcquisitionGeometry.create_Cone3D(source_position= [0,0,0],# TODO: check this
                detector_position = [0, self._metadata['source_detector_distance'], 0], rotation_axis_position= [0, self._metadata['source_detector_distance']-self._metadata['sample_detector_distance'], 0],
                units = self._metadata['distance_units'],
                ) \
                .set_panel([horizontal, vertical], 
                            pixel_size=[self._metadata['pixel_size_x'], 
                                        self._metadata['pixel_size_y']]) \
                .set_angles(self._metadata['angles'], angle_unit=self._metadata['angle_units'])
            if self.channels:
                self._acquisition_geometry.set_channels(self._metadata['num_channels'])
        else:
            self._acquisition_geometry = AcquisitionGeometry.create_Cone2D( source_position= [0,0],# TODO: check this
                detector_position= [0, self._metadata['source_detector_distance']], rotation_axis_position= [0,self._metadata['source_detector_distance']-self._metadata['sample_detector_distance']],
                units = self._metadata['distance_units']
            )\
                .set_panel(horizontal, 
                            pixel_size=self._metadata['pixel_size_x']) \
                .set_angles(self._metadata['angles'], angle_unit=self._metadata['angle_units'])
            if self.channels:
                self._acquisition_geometry.set_channels(self._metadata['num_channels'])
            
        self._acquisition_geometry.dimension_labels = self._dimension_labels

    def _read_data(self, dtype=np.float32, roi=(slice(None),slice(None),slice(None))):
        
        if isinstance(self._dataset_path,(tuple,list)):
            ad = []
            start = 0
            for x in self._dataset_path:
                data =  HDF5_utilities.read(self._file_name, x, 
                                    source_sel=tuple(roi), dtype=dtype)
                length = data.shape[0]
                ad[start:(start+length)] = data
                start +=length

                del data
        else:   

            ad =  HDF5_utilities.read(self._file_name, self._dataset_path, 
                                    source_sel=tuple(roi), dtype=dtype)
        

        # self._data_reader.dtype = dtype
        # self._data_reader.set_roi(roi)
        if self.flatfield_path is not None:
            flatfield = HDF5_utilities.read(self.norm_filename, self.flatfield_path)
            try:
                num_repeats = len(flatfield)
            except:
                num_repeats = 1
            geom = self._acquisition_geometry.copy()
            geom.set_angles(np.ones(num_repeats))
            self.flatfield = AcquisitionData(flatfield, geometry=geom)

        if self.darkfield_path is not None:
            darkfield = HDF5_utilities.read(self.norm_filename, self.darkfield_path)
            try:
                num_repeats = len(darkfield)
            except:
                num_repeats = 1
            geom = self._acquisition_geometry.copy()
            geom.set_angles(np.ones(num_repeats))
            self.darkfield = AcquisitionData(darkfield, geometry=geom)

        return ad




    def get_raw_data(self):
        """
        Get the raw data array if not already in memory
        """
        return self._data_handle.get_data(dtype=None, roi=None)


    def _get_data_array(self, selection):
        """
        Method to read an roi of the data from disk and return an `numpy.ndarray`.

        selection is a tuple of slice objects for each dimension
        """
        return self._data_handle.get_data(dtype=np.float32, roi=selection,)



    def _get_data(self, projection_indices=None):
        """
        Method to read the data from disk and bin as requested. Returns an `numpy.ndarray`

        if projection_indices is None will use based on set_angles
        """



        # if override default (this is used by preview)
        if projection_indices is None:
            indices = self._indices
        else:
            indices = projection_indices

        if indices is None: 
            selection = (slice(0, self.full_geometry.num_projections), *self._panel_crop)
            output_array = self._get_data_array(selection)

        elif isinstance(indices,(range,slice)):   
            selection = (slice(indices.start, indices.stop, indices.step),*self._panel_crop)
            output_array = self._get_data_array(selection)

        elif isinstance(indices, int):
            selection = (slice(indices, indices+1),*self._panel_crop)
            output_array = self._get_data_array(selection)

        elif isinstance(indices,(list,np.ndarray)):

            # need to make this shape robust
            output_array = np.empty((len(indices), *self.full_geometry.shape[1::]), dtype=np.float32)

            i = 0
            while i < len(indices):

                ind_start = i
                while ((i+1) < len(indices)) and (indices[i] + 1 == indices[i+1]):
                    i+= 1

                i+=1
                selection = (slice(indices[ind_start], indices[ind_start] + i-ind_start),*self._panel_crop)
                output_array[ind_start:ind_start+i-ind_start,:,:] = self._get_data_array(selection)


        else:
            raise ValueError("Nope")

        #what if sliced and reduced dimensions?
        proj_unbinned = DataContainer(output_array,False, self.full_geometry.dimension_labels)

        if self._bin:
            binner = Binner(roi={'vertical':(None,None,self._bin_roi[0]),'horizontal':(None,None,self._bin_roi[1])})
            binner.set_input(proj_unbinned) 
            output_array = binner.get_output().array

        return output_array.squeeze()


    def _parse_crop_bin(self, arg, length):
        """
        Method to parse the input roi as a int or tuple (start, stop, step) perform checks and return values
        """
        crop = slice(None,None)
        step = 1

        if arg is not None:
            if isinstance(arg,int):
                crop = slice(arg, arg+1)
                step = 1
            elif isinstance(arg,tuple):
                slice_obj = slice(*arg)
                crop = slice(slice_obj.start, slice_obj.stop)

                if slice_obj.step is None:
                    step = 1
                else:
                    step = slice_obj.step
            else:
                raise TypeError("Expected input to be an int or tuple. Got {}".format(arg))
        
        
        range_new = range(0, length)[crop]

        if len(range_new)//step < 1:
            raise ValueError("Invalid ROI selection. Cannot")  
        
        return crop, step


    def set_panel_roi(self, vertical=None, horizontal=None):
        """
        TODO: needs fixing for 2D data
        
        Method to configure the ROI of data to be returned as a CIL object.

        horizontal: takes an integer for a single slice, a tuple of (start, stop, step)
        vertical: tuple of (start, stop, step), or `vertical='centre'` for the centre slice

        If step is greater than 1 pixels will be averaged together.
        """

        if vertical == 'centre':
            dim = self.full_geometry.dimension_labels.index('vertical')
            
            centre_slice_pos = (self.full_geometry.shape[dim]-1) / 2.
            ind0 = int(np.floor(centre_slice_pos))

            w2 = centre_slice_pos - ind0
            if w2 == 0:
                vertical=(ind0, ind0+1, 1)
            else:
                vertical=(ind0, ind0+2, 2)
        if vertical is not None:
            crop_v, step_v = self._parse_crop_bin(vertical, self.full_geometry.pixel_num_v)
        if horizontal is not None:
            crop_h, step_h = self._parse_crop_bin(horizontal, self.full_geometry.pixel_num_h)
        
        if step_v > 1 or step_h > 1:
            self._bin = True
        else:
            self._bin = False

        self._bin_roi = (step_v, step_h)
        self._panel_crop = (crop_v, crop_h)


    def set_angles(self, indices=None):
        """
        Method to configure the angular indices to be returned as a CIL object.

        indices: takes an integer for a single projections, a tuple of (start, stop, step), 
        or a list of indices.

        If step is greater than 1 pixels the data will be sliced. i.e. a step of 10 returns 1 in 10 projections.
        """      

        if indices is not None:
            if isinstance(indices,tuple):
                indices = slice(*indices)
            elif isinstance(indices,(list,np.ndarray)):
                indices = indices
            elif isinstance(indices,int):
                indices = [indices]
            else:
                raise TypeError("Expected input to be an int, tuple or list. Got {}".format(indices))
        
            try:
                angles = self.full_geometry.angles[(indices)]

            except IndexError:
                raise ValueError("Out of range")
            
            if angles.size < 1:
                raise ValueError(") projections selected. Please select at least 1 angle")
        self._indices = indices
        

    def reset(self):
        """
        Resets the configured ROI and angular indices to the full dataset
        """
        # range or list object for angles to process, defaults to None
        self._indices = None

        # slice in each dimension, initialised to none
        self._panel_crop = (slice(None),slice(None))

        # number of pixels to bin in each dimension
        self._bin_roi = (1,1)

        # boolean if binned
        self._bin = False


    def preview(self, initial_angle=0):
        """
        Displays two normalised projections approximately 90 degrees apart.

        This respects the configured ROI and angular indices.

        Parameters
        ----------
        initial_angle: float
            Set the angle of the 1st projection in degrees
        """

        ag = self.get_geometry()
        angles = ag.angles.copy()


        if ag.config.angles.angle_unit == 'degree':
            ang1 = initial_angle
            ang2 = ang1+90

            #angles in range 0->360
            for i, a in enumerate(angles):
                while a < 0:
                    a += 360
                while a >= 360:
                    a -= 360
                angles[i] = a

        if ag.config.angles.angle_unit == 'radian':
            ang1 = initial_angle
            ang2 = ang1+np.pi

            #angles in range 0->2*pi
            for i, a in enumerate(angles):
                while a < 0:
                    a += 2 * np.pi
                while a >= 2*np.pi:
                    a -= 2 * np.pi
                angles[i] = a


        idx_1 = np.argmin(np.abs(angles-ang1))
        idx_2 = np.argmin(np.abs(angles-ang2))

        ag.set_angles([angles[idx_1], angles[idx_2]])
        
        # overide projectsions to be read
        data = self._get_data(projection_indices=[idx_1,idx_2])
        show2D(data, slice_list=[0,1], title= [str(angles[idx_1])+ ag.config.angles.angle_unit, str(angles[idx_2]) +ag.config.angles.angle_unit],origin='upper-left')


    def get_geometry(self):
        """
        Method to retrieve the geometry describing your data.

        This respects the configured ROI and angular indices.

        Returns
        -------
        AcquisitionGeometry
            Returns an AcquisitionGeometry describing your system.
        """

        ag = self.full_geometry.copy()

        if isinstance(self._indices,slice):
            ag.config.angles.angle_data = ag.angles[(self._indices)]
        elif isinstance(self._indices,list):
            ag.config.angles.angle_data = np.take(ag.angles, list(self._indices))
        ds_metadata = HDF5_utilities.get_dataset_metadata(self._file_name, 
                                                          self._dataset_path)
        if len(ds_metadata['shape']) > 2+self.channels: #TODO: need a shape method? 
            #slice and bin geometry
            roi = { 'horizontal':(self._panel_crop[1].start, self._panel_crop[1].stop, self._bin_roi[1]),
                    'vertical':(self._panel_crop[0].start, self._panel_crop[0].stop, self._bin_roi[0]),
            }

            return Binner(roi)(ag)
        else:
            #TODO: needs fixing for 2D data to allow for roi and binning
            return ag


    def read(self):
        """
        Method to retrieve the data .

        This respects the configured ROI and angular indices.

        Returns
        -------
        AcquisitionData
            Returns an AcquisitionData containing your data and AcquisitionGeometry.
        """

        geometry = self.get_geometry()
        data = self._get_data()
        return AcquisitionData(data, False, geometry)
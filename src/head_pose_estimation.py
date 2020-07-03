

# %%
'''
Import all the required modules
'''
from model import Model
import cv2
import numpy as np
import logging

# %%
'''
Define ModelHeadPoseEstimation class
'''

class ModelHeadPoseEstimation(Model):

    '''
    Class for the head pose estimation model
    https://docs.openvinotoolkit.org/2020.1/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
    '''

    __MODEL_NAME = 'head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'


    def __init__(self, model_name=__MODEL_NAME, device='CPU', precision='FP32'):

        '''
        Loads selected version of the head pose estimation model to the selected
        device

        Note:
            If `device` is set to ``MYRIAD`` or ``GPU`` then `precision` is 
            automatically set to ``FP16`` 

        Args:
            model_name (:obj:`str`, optional): name of the model to load
            device (:obj:`str`, optional): device to infer on
            precision (:obj:`str`, optional): precision of the model to load
        '''

        if precision is not 'FP32':
            model_name.replace('FP32', precision)
        if device.startswith('MYRIAD') or device.startswith('MULTI:MYRIAD') or \
            device.startswith('HETERO:MYRIAD') or device.startswith('GPU') or \
                device.startswith('MULTI:GPU') or device.startswith('HETERO:GPU'):
                model_name = 'head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001'
        super().__init__(self._MODEL_PATH + model_name, device)


    def preprocess_inputs(self, inputs):

        '''
        Resize input image to the size of [3x60x60] and appends it to the list

        Args:
            inputs (ndarray): image with human face

        Returns:
            list: list of resized images 
        '''

        self.input = inputs
        input = cv2.resize(self.input, (self._input_shape[3], \
            self._input_shape[2]), interpolation = cv2.INTER_AREA)
        result = []
        result.append(np.moveaxis(input, -1, 0))
        return result


    def preprocess_outputs(self, outputs):

        '''
        Extract head pose euler angles from the list

        Args:
            outputs (list): list of head pose euler angles

        Returns:
            ndarray: vector with head pose euler angles
        '''
        self.debug = np.asarray(outputs).ravel()
        logging.info('Head pose: ' + str(self.debug))
        return self.debug

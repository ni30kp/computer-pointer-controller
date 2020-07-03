

# %%
'''
Import all the required modules
'''

from model import Model
import numpy as np
import cpuinfo
import logging

# %%
'''
Define ModelGazeEstimation class
'''

class ModelGazeEstimation(Model):

    '''
    Class for the gaze estmation model
    https://docs.openvinotoolkit.org/2020.1/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
    '''

    __MODEL_NAME = 'gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'


    def __init__(self, model_name=__MODEL_NAME, device='CPU', precision='FP32'):

        '''
        Loads selected version of the gaze estimation model to the selected
        device

        Note:
            If `device` is set to ``MYRIAD`` or ``GPU`` then `precision` is 
            automatically set to ``FP16`` 
            If `device` is set to ``CPU`` and `precision` is set to 
            ``FP32-INT8`` and detected processor is lower then 6th generation 
            then then `precision` is automatically set to `FP32`. Model of
            ``FP32-INT8`` precision triggers ``Illegal instruction`` crash.

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
            model_name = 'gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002'
        if (device.startswith('CPU') or device.startswith('MULTI:CPU') or \
            device.startswith('HETERO:CPU')) and (precision is 'FP32-INT8') \
            and cpuinfo.get_cpu_info()['extended_model'] < 6:
            model_name = 'gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'
        super().__init__(self._MODEL_PATH + model_name, device)


    def preprocess_inputs(self, inputs):

        '''
        Reorders list of inputs 

        Args:
            inputs (list): left and right eye images of size [3x60x60] and head 
                pose vector with euler angles

        Return:
            list: reordered list of inputs
        '''

        result = []
        result.append(inputs[2])
        result.append(np.moveaxis(inputs[0], -1, 0))
        result.append(np.moveaxis(inputs[1], -1, 0))
        return result


    def preprocess_outputs(self, outputs):

        '''
        Unpacks gaze vector from the list

        Args:
            outputs (list): list with gaze vector

        Returns:
            ndarray: gaze vector
        '''

        self.debug = outputs[0][0]
        logging.info('Gaze: ' + str(self.debug))
        return self.debug

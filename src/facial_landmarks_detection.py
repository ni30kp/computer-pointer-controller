

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
Define ModelFacialLandmarksDetection class
'''

class ModelFacialLandmarksDetection(Model):

    '''
    Class for the facial landmarks detection model
    https://docs.openvinotoolkit.org/2020.1/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
    '''

    __MODEL_NAME = 'landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'


    def __init__(self, model_name=__MODEL_NAME, device='CPU', precision='FP32'):

        '''
        Loads selected version of the facial landmarks detection model to the
        selected device

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
            model_name = 'landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009'
        super().__init__(self._MODEL_PATH + model_name, device)


    def preprocess_inputs(self, inputs):

        '''
        Resizes input image to the size of [3x48x48] and appends it to the list

        Args:
            inputs (ndarray): input image with the human face
        
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
        Extracts images of facial landmarks related to left eye and right eye
        and appends it to the list

        Args:
            outputs (list): list of facial landmarks

        Returns:
            list: list of images with left and right eye
        '''

        output = outputs[0].ravel()
        logging.info('Facial landmarks: '+ str(output))
        result = []
        self.debug = [(int(output[0]*self.input.shape[1]), \
            int(output[1]*self.input.shape[0])), \
                (int(output[2]*self.input.shape[1]), \
                    int(output[3]*self.input.shape[0]))]
        image = self.input[self.debug[0][1]-30:self.debug[0][1]+30,\
                self.debug[0][0]-30:self.debug[0][0]+30]
        result.append(image)
        image = self.input[self.debug[1][1]-30:self.debug[1][1]+30,\
                self.debug[1][0]-30:self.debug[1][0]+30]
        result.append(image)
        return result

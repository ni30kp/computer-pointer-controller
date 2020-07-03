

# %%
'''
Import all the required modules
'''
from model import Model
import cv2
import numpy as np
import logging

#%% 
'''
Define ModelFaceDetection class
'''

class ModelFaceDetection(Model):

    '''
    Class for the face detection model
    https://docs.openvinotoolkit.org/2020.1/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html   
    https://docs.openvinotoolkit.org/2020.1/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html
    '''

    __MODEL_NAME = 'face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
    

    def __init__(self, model_name=__MODEL_NAME, device='CPU', threshold = 0.9):

        '''
        Loads selected version of the face detection model to the selected
        device

        Note:
            If `device` is set to ``MYRIAD`` or ``GPU`` then `precision` is 
            automatically set to ``FP16`` 

        Args:
            model_name (:obj:`str`, optional): name of the model to load
            device (:obj:`str`, optional): device to infer on
            threshold (:obj:`float`, optional): confidence level threshold for 
                face detection
        '''

        if device.startswith('MYRIAD') or device.startswith('MULTI:MYRIAD') or \
            device.startswith('HETERO:MYRIAD') or device.startswith('GPU') or \
                device.startswith('MULTI:GPU') or device.startswith('HETERO:GPU'):
            model_name = 'face-detection-adas-0001/FP16/face-detection-adas-0001'
        
        super().__init__(self._MODEL_PATH + model_name, device)
        self.threshold = threshold


    def preprocess_inputs(self, inputs):

        '''
        Resizes the input image to the size of [3x384x672] and appends it to
        the list

        Args:
            inputs (ndarray): input image

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
        Crops the input image to the specified size detected by the model +60

        Args:
            outputs (list): list of faces localisation coordinates within an 
                image

        Returns:
            list: list of cropped images        
        '''

        self.debug = []
        for data in outputs[0][0][0]:
            if data[2] > self.threshold and data[1] == 1:
                image = self.input[int(data[4]*self.input.shape[0])-30: \
                    int(data[6]*self.input.shape[0])+30, \
                        int(data[3]*self.input.shape[1])-30: \
                            int(data[5]*self.input.shape[1])+30]
                logging.info('Image size: '+str(image.shape))
                self.debug.append(image)
        return self.debug

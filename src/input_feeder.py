

# %%
'''
Import all the required modules
'''
import cv2
from numpy import ndarray

# %%
'''
Define InputFeeder class
'''

class InputFeeder:

    '''
    Input feeder class is responsible of delivery of the input images
    '''

    def __init__(self, input):

        '''
        Initializes images capture from image or video file or from the camera

        Args:
            input (str): path to the image or video file or camera pipeline
        '''

        self.cap=cv2.VideoCapture(input)



    def next_batch(self):

        '''
        Returns the next image from either a video file or camera.
        '''

        while self.cap.isOpened():
            flag, frame = self.cap.read()
            yield frame


    def close(self):

        '''
        Closes the VideoCapture
        '''

        self.cap.release()



# %%
'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
# %%
'''
Import all the required modules
'''

import pyautogui

# %%
'''
Define MouseController class
'''

class MouseController:

    '''
    Mouse controller class enables to control computer pointer
    '''

    def __init__(self, precision, speed):

        '''
        Initialize precision and speed of mouse controller

        Args:
            precision (str): precision of the mouse controller: ``low``,
                ``medium``, ``high``
            speed (str): speed of the mouse controller: ``slow``, ``medium``,
                ``fast``
        '''        

        precision_dict = {'high':100, 'low':1000, 'medium':500}
        speed_dict = {'fast':0, 'slow':10, 'medium':5}
        self.precision = precision_dict[precision]
        self.speed = speed_dict[speed]
        pyautogui.FAILSAFE = False


    def move(self, x, y):
                            
        '''
        Performs relative movement of the computer pointer

        Args:
            x (int): movement in x direction 
            y (int): movement in y direction
        '''      

        pyautogui.moveRel(x*self.precision, -1*y*self.precision, \
            duration=self.speed)


    def center(self):

        '''
        Sets computer pointer to the center of the screen
        '''        

        size = pyautogui.size()
        center = pyautogui.center((0, 0, size[0], size[1]))
        pyautogui.moveTo(center[0], center[1])

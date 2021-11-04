import numpy as np
from Visual_odometry._globals import *
from Visual_odometry.visual_odometry import VisualOdometry


class visual_odometry(object):
    '''Ratslam implementation.

    The ratslam is divided into 4 modules: visual odometry, view cells, pose 
    cells, and experience map. This class also store the odometry and pose 
    cells activation in order to plot them.
    '''

    def __init__(self):
        '''Initializes the ratslam modules.'''

        self.visual_odometry = VisualOdometry()

        # TRACKING -------------------------------
        x, y, th = self.visual_odometry.odometry       # x： x轴方向上的线速度分量， y: y轴方向上的线速度分量， th: 角度
        self.odometry = [[x], [y], [th]]
        
        # ----------------------------------------

    def digest(self, img):
        '''Execute a step of ratslam algorithm for a given image.

        :param img: an gray-scale image as a 2D numpy array.
        '''
        vtrans, vrot = self.visual_odometry(img)                           # 通过图像计算出线速度vtrans, 和角速度vrot

        # TRACKING -------------------------------
        x, y, th = self.visual_odometry.odometry
        self.odometry[0].append(x)
        self.odometry[1].append(y)
        self.odometry[2].append(th)
        


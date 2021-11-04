import numpy as np
from Visual_odometry._globals import *
import cv2

class VisualOdometry(object):
    '''Visual Odometry Module.'''

    def __init__(self):
        '''Initializes the visual odometry module.'''
        self.old_vtrans_template = np.zeros(IMAGE_ODO_X_RANGE.stop)
        self.old_vrot_template = np.zeros(IMAGE_ODO_X_RANGE.stop)
        self.odometry = [0., 0., np.pi/2]

    def _create_template(self, subimg):
        '''Compute the sum of columns in subimg and normalize it.

        :param subimg: a sub-image as a 2D numpy array.
        :return: the view template as a 1D numpy array.
        '''
        x_sums = np.sum(subimg, 0)                              # 对每列进行求和,现在的 size 是 n
        avint = np.sum(x_sums, dtype=np.float32)/x_sums.size    # 相当于求了一个平均值
        return x_sums/avint                                     # 每列求和之后，除以平均值

    def __call__(self, img):
        '''Execute an interation of visual odometry.

        :param img: the full gray-scaled image as a 2D numpy array.
        :return: the deslocation and rotation of the image from the previous 
                 frame as a 2D tuple of floats.
        '''
        subimg = img[IMAGE_VTRANS_Y_RANGE, IMAGE_ODO_X_RANGE]   # [slice(270, 430), slice(180+15, 460+15)]  即从图片中间截出了一块
        template = self._create_template(subimg)                # 每列求和之后，除以平均值 280

        # VTRANS
        offset, diff = compare_segments(
            template,                                           # 新的template
            self.old_vtrans_template,                           # 旧的template
            VISUAL_ODO_SHIFT_MATCH                              # VISUAL_ODO_SHIFT_MATCH  = 140
        )
        vtrans = diff*VTRANS_SCALE                              # 求出的线速度  VTRANS_SCALE  = 100

        if vtrans > 10: 
            vtrans = 0

        self.old_vtrans_template = template                     # 将现在的template赋给旧的template

        # VROT
        subimg = img[IMAGE_VROT_Y_RANGE, IMAGE_ODO_X_RANGE]
        template = self._create_template(subimg)

        offset, diff = compare_segments(
            template, 
            self.old_vrot_template,
            VISUAL_ODO_SHIFT_MATCH
        )
        vrot = offset*(50./img.shape[1])*np.pi/180
        self.old_vrot_template = template

        # Update raw odometry
        self.odometry[2] += vrot 
        self.odometry[0] += vtrans*np.cos(self.odometry[2])
        self.odometry[1] += vtrans*np.sin(self.odometry[2])

        return vtrans, vrot
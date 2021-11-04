import numpy as np
import itertools

def min_delta(d1, d2, max_):
    delta = np.min([np.abs(d1-d2), max_-np.abs(d1-d2)])
    return delta

def clip_rad_180(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

def clip_rad_360(angle):
    while angle < 0:
        angle += 2*np.pi
    while angle >= 2*np.pi:
        angle -= 2*np.pi
    return angle

def signed_delta_rad(angle1, angle2):
    dir = clip_rad_180(angle2 - angle1)
    
    delta_angle = abs(clip_rad_360(angle1) - clip_rad_360(angle2))
    
    if (delta_angle < (2*np.pi-delta_angle)):
        if (dir>0):
            angle = delta_angle
        else:
            angle = -delta_angle
    else: 
        if (dir>0):
            angle = 2*np.pi - delta_angle
        else:
            angle = -(2*np.pi-delta_angle)
    return angle


def create_pc_weights(dim, var):       # dim: PC_W_E_DIM = 7 ; PC_W_I_DIM = 5     var: PC_W_E_VAR = 1 ; PC_W_I_VAR = 2   
                                       # 即：dim表示这个高斯分布所能影响的范围， var表示这个高斯分布的形状，值越大，形状越胖
    dim_center = int(np.floor(dim/2.))
    
    weight = np.zeros([dim, dim, dim])
    for x, y, z in itertools.product(range(dim), range(dim), range(dim)):
        dx = -(x-dim_center)**2
        dy = -(y-dim_center)**2
        dz = -(z-dim_center)**2
        weight[x, y, z] = 1.0/(var*np.sqrt(2*np.pi))*np.exp((dx+dy+dz)/(2.*var**2))   # weight 矩阵就是一个三维高斯分布

    weight = weight/np.sum(weight)   # np.sum(weight)求得是 weight 全部元素的和, 此操作是在进行归一化
    return weight

# @profile
def compare_segments(seg1, seg2, slen):  # seg1 = [1, 280],  seg2 = [1, 280],  slen = 140
    cwl = seg1.size                      # cwl = 280

    mindiff = 1e10
    minoffset = 0

    diffs = np.zeros(slen)               # diffs = [0, 0, ...,  0] : 140

    for offset in range(slen+1):
        e = (cwl-offset)

        cdiff = np.abs(seg1[offset:cwl] - seg2[:e])
        cdiff = np.sum(cdiff)/e

        if cdiff < mindiff:
            mindiff = cdiff
            minoffset = offset

        cdiff = np.abs(seg1[:e] - seg2[offset:cwl])
        cdiff = np.sum(cdiff)/e

        if cdiff < mindiff:
            mindiff = cdiff
            minoffset = -offset

    return minoffset, mindiff

# CONSTANTS AND ALGORITHM PARAMETERS ==========================================
# NOTE: it is need a refactoring to set these variables as a model parameter
PC_VT_INJECT_ENERGY     = 0.1
PC_DIM_XY               = 61
PC_DIM_TH               = 36
PC_W_E_VAR              = 1
PC_W_E_DIM              = 7
PC_W_I_VAR              = 2
PC_W_I_DIM              = 5
PC_GLOBAL_INHIB         = 0.00002
PC_W_EXCITE             = create_pc_weights(PC_W_E_DIM, PC_W_E_VAR)   # create_pc_weights(7, 1)
PC_W_INHIB              = create_pc_weights(PC_W_I_DIM, PC_W_I_VAR)   # create_pc_weights(5, 2)
PC_W_E_DIM_HALF         = int(np.floor(PC_W_E_DIM/2.))   # 3
PC_W_I_DIM_HALF         = int(np.floor(PC_W_I_DIM/2.))   # 2
PC_C_SIZE_TH            = (2.*np.pi)/PC_DIM_TH           # 0.17453292519943295   即将2 pi 分成36份， 每份是0.1745...
PC_E_XY_WRAP            = list(range(PC_DIM_XY-PC_W_E_DIM_HALF, PC_DIM_XY)) + list(range(PC_DIM_XY)) + list(range(PC_W_E_DIM_HALF))  # list(range(61-3, 61)) + list(range(61)) + list(range(3))
PC_E_TH_WRAP            = list(range(PC_DIM_TH-PC_W_E_DIM_HALF, PC_DIM_TH)) + list(range(PC_DIM_TH)) + list(range(PC_W_E_DIM_HALF))  # list(range(36-3, 36)) + list(range(36)) + list(range(3))
PC_I_XY_WRAP            = list(range(PC_DIM_XY-PC_W_I_DIM_HALF, PC_DIM_XY)) + list(range(PC_DIM_XY)) + list(range(PC_W_I_DIM_HALF))  # list(range(61-2, 61)) + list(range(61)) + list(range(2))
PC_I_TH_WRAP            = list(range(PC_DIM_TH-PC_W_I_DIM_HALF, PC_DIM_TH)) + list(range(PC_DIM_TH)) + list(range(PC_W_I_DIM_HALF))  # list(range(36-2, 36)) + list(range(36)) + list(range(2))          
PC_XY_SUM_SIN_LOOKUP    = np.sin(np.multiply(list(range(1, PC_DIM_XY+1)), (2*np.pi)/PC_DIM_XY))  # 从sin(2*pi/61) ~ sin(2*pi)
PC_XY_SUM_COS_LOOKUP    = np.cos(np.multiply(list(range(1, PC_DIM_XY+1)), (2*np.pi)/PC_DIM_XY))  # 从cos(2*pi/61) ~ cos(2*pi)
PC_TH_SUM_SIN_LOOKUP    = np.sin(np.multiply(list(range(1, PC_DIM_TH+1)), (2*np.pi)/PC_DIM_TH))  # 从sin(2*pi/36) ~ sin(2*pi)
PC_TH_SUM_COS_LOOKUP    = np.cos(np.multiply(list(range(1, PC_DIM_TH+1)), (2*np.pi)/PC_DIM_TH))  # 从cos(2*pi/36) ~ cos(2*pi)
PC_CELLS_TO_AVG         = 3
PC_AVG_XY_WRAP          = list(range(PC_DIM_XY-PC_CELLS_TO_AVG, PC_DIM_XY)) + list(range(PC_DIM_XY)) + list(range(PC_CELLS_TO_AVG))  # list(range(61-3, 61)) + list(range(61)) + list(range(3))
PC_AVG_TH_WRAP          = list(range(PC_DIM_TH-PC_CELLS_TO_AVG, PC_DIM_TH)) + list(range(PC_DIM_TH)) + list(range(PC_CELLS_TO_AVG))  # list(range(36-3, 36)) + list(range(36)) + list(range(3))
IMAGE_Y_SIZE            = 640
IMAGE_X_SIZE            = 480
IMAGE_VT_Y_RANGE        = slice((480/2 - 80 - 40), (480/2 + 80 - 40))    # slice(120, 280)
IMAGE_VT_X_RANGE        = slice((640/2 - 280 + 15), (640/2 + 280 + 15))  # slice(55, 615)
IMAGE_VTRANS_Y_RANGE    = slice(270, 430)
IMAGE_VROT_Y_RANGE      = slice(75, 240)
IMAGE_ODO_X_RANGE       = slice(180+15, 460+15)                          # slice(195, 475)
VT_GLOBAL_DECAY         = 0.1
VT_ACTIVE_DECAY         = 1.0
VT_SHIFT_MATCH          = 20
VT_MATCH_THRESHOLD      = 0.09
EXP_DELTA_PC_THRESHOLD  = 1.0
EXP_CORRECTION          = 0.5
EXP_LOOPS               = 100
VTRANS_SCALE            = 100
VISUAL_ODO_SHIFT_MATCH  = 140
ODO_ROT_SCALING         = np.pi/180./7.
POSECELL_VTRANS_SCALING = 1./10.
# =============================================================================

ANGLE_data_path = 'E:/datasets/ANGLE/Ch2_001/'
ANGLE_data_name = 'final_example.csv'
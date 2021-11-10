from torch.nn.modules.activation import Sigmoid
import _globals
import csv
from brian2 import *
import brian2 as b2
import torch.nn as nn
import torch

#---------------------------------------------
# parameters
#---------------------------------------------

v_rest_e = -65. * b2.mV
v_rest_i = -60. * b2.mV
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV
v_thresh_i = -40. * b2.mV
refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

tc_pre_ee = 20*b2.ms
tc_post_1_ee = 20*b2.ms
tc_post_2_ee = 40*b2.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1


def create_Weight_Matrix(weight_M, dim_0, dim_1):

    Weight_Matrix = np.zeros((dim_0, dim_1))
    Weight_Matrix[weight_M[:,0], weight_M[:,1]] = weight_M[:,3]
    return Weight_Matrix

#----------------------------------------
weight_path = 'D:/python_pro/pro_slam_odometry_git'
weight_save_path = 'D:/python_pro/pro_slam_odometry_git/'
#----------------------------------------

tc_theta = 1e7 * b2.ms
theta_plus_e = 0.05 * b2.mV
scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b2.mV
v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
v_thresh_i_str = 'v>v_thresh_i'
v_reset_i_str = 'v=v_reset_i'

device = 'cpu'

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(400, 512),
            nn.Sigmoid(),
            nn.Linear(512, 400),
            nn.Sigmoid(),
            nn.Linear(400, 100),
            nn.Sigmoid(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        return self.net(x)




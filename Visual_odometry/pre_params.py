import _globals
import csv
from brian2 import *
import brian2 as b2
# from Visual_odometry.Spiking_model import *


#---------------------------------------------
# parameters
#---------------------------------------------
update_interval = 3000
n_e = 400                     # 兴奋层神经元个数
n_i = 400                     # 抑制层神经元个数
n_input = 280

v_rest_e = -65. * b2.mV
v_rest_i = -60. * b2.mV
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV
v_thresh_i = -40. * b2.mV
refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input']
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b2.ms,10*b2.ms)
delay['ei_input'] = (0*b2.ms,5*b2.ms)
input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b2.ms
tc_post_1_ee = 20*b2.ms
tc_post_2_ee = 40*b2.ms
nu_ee_pre =  0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4
ee_STDP_on = True

#----------------------------------------
weight_path = 'D:/python_pro/pro_slam_odometry_git'
#----------------------------------------

neuron_weights = np.zeros((n_e, 10))    # 表示400个神经元，每个神经元对每个类的敏感程度

# scr_e代表的是复位操作方程

tc_theta = 1e7 * b2.ms
theta_plus_e = 0.05 * b2.mV
scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b2.mV
v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
v_thresh_i_str = 'v>v_thresh_i'
v_reset_i_str = 'v=v_reset_i'

# 兴奋层神经元方程
neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'


# 抑制层神经元方程
neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                   : 1
        '''

# 突触连接处的STDP算法方程        
eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# pre methods for the model
#-------------------------------------------------------------------------------
ending = ''

# def get_angle_data(csv_name, bTrue = True):
#     '''
#     this method is used to read the angle data of the running car
#     '''
#     with open(ANGLE_data_path + ANGLE_data_name) as f:
#         f_csv = csv.reader(f)
#         headers = next(f_csv)

#         for row in f_csv:
#             print(row[1])




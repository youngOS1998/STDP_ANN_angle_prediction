import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy
from struct import unpack
from brian2 import *
import brian2 as b2
from brian2tools import *
import matplotlib.pyplot as plt 
import csv
from Visual_odometry._globals import *

#---------------------------------------------
# 以下是LIF神经元模型和STDP突触模型
#---------------------------------------------

#---------------------------------------------
# parameters
#---------------------------------------------
update_interval = 3000
n_e = 400                     # 兴奋层神经元个数

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
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

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


# 构建兴奋层和抑制层神经元组

b2.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))

# 此是兴奋层, 400个神经元
# population_names = ['A']
neuron_groups['e'] = b2.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e_str, refractory= refrac_e, reset= scr_e, method='euler')

# 此是抑制层， 400个神经元
neuron_groups['i'] = b2.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i_str, refractory= refrac_i, reset= v_reset_i_str, method='euler')

# population_names = ['A']

#------------------------------------------------------------------------------
# create network population and recurrent connections                    'AeAi'
#------------------------------------------------------------------------------
for subgroup_n, name in enumerate(population_names): # population_names = ['A']
    print ('create neuron group', name)

    neuron_groups[name+'e'] = neuron_groups['e'][subgroup_n*n_e:(subgroup_n+1)*n_e]  # 400 neurons for 'Ae'
    neuron_groups[name+'i'] = neuron_groups['i'][subgroup_n*n_i:(subgroup_n+1)*n_e]  # 400 neurons for 'Ai'

    # v_rest_e = -65. * b2.mV
    # set the initial potential for excitatory layer and inhibitory layer
    # for ex layer: -105 mv
    # for in layer: -105 mv

    neuron_groups[name+'e'].v = v_rest_e - 40. * b2.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b2.mV

    neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b2.mV

    print ('create recurrent connections')

    #----------------------------------------------------------------------------------------------------------------------------
    # 下面这个循环分别创建从Ae到Ai, 和Ai到Ae的连接：

    ###  Ae to Ai:
    # 在weightMatrix这个矩阵里面是只连接对应的抑制神经单元，其他的连接权重都为0
    # model = 'w : 1'
    # pre = 'ge_post += w'
    # post = ' '

    ###  Ai to Ae:
    # 在weightMatrix这个矩阵里面是抑制单元只连接 除 对应兴奋单元的其他兴奋单元，即对应兴奋神经单元的连接权重为0，其他连接神经单元权重为10.4
    # model = 'w : 1'
    # pre = 'ge_post += w'
    # post = ' '
    #----------------------------------------------------------------------------------------------------------------------------

    for conn_type in recurrent_conn_names:  # recurrent_conn_names = ['ei', 'ie']
        connName = name+conn_type[0]+name+conn_type[1] # 'AeAi' 'AiAe'
        weightMatrix = get_matrix_from_file(weight_path + '../random/' + connName + ending + '.npy')
        model = 'w : 1'
        pre = 'g%s_post += w' % conn_type[0]
        post = ''
        if ee_STDP_on:
            if 'ee' in recurrent_conn_names:
                model += eqs_stdp_ee
                pre += '; ' + eqs_stdp_pre_ee
                post = eqs_stdp_post_ee
        
        # connections = {} 中有三个，一个是'AeAi'的突触连接, 一个是'AiAe'的突触连接，后面还会增加一个'XeAe'的突触连接
        connections[connName] = b2.Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                    model=model, on_pre=pre, on_post=post)
        connections[connName].connect(True) # all-to-all connection
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]     # 从事先存好的矩阵数据中读取权重信息

    print ('create monitors for', name)
    rate_monitors[name+'e'] = b2.PopulationRateMonitor(neuron_groups[name+'e'])   # Ae这个神经元组的PopulationRateMonitor()
    rate_monitors[name+'i'] = b2.PopulationRateMonitor(neuron_groups[name+'i'])   # Ai这个神经元组的PopulationRateMonitor()
    spike_counters[name+'e'] = b2.SpikeMonitor(neuron_groups[name+'e'])

    if record_spikes:  # True
        spike_monitors[name+'e'] = b2.SpikeMonitor(neuron_groups[name+'e'])       # Ae这个神经元组的脉冲记录
        spike_monitors[name+'i'] = b2.SpikeMonitor(neuron_groups[name+'i'])       # Ai这个神经元组的脉冲记录

#------------------------------------------------------------------------------
# create input population and connections from input populations         'XeAe'
#------------------------------------------------------------------------------
pop_values = [0,0,0]
for i,name in enumerate(input_population_names):                                  # input_population_names = 'X'
    input_groups[name+'e'] = b2.PoissonGroup(n_input, 0*Hz)                       # input_groups['Xe'] = b2.PoissonGroup(784, 0*Hz)
    rate_monitors[name+'e'] = b2.PopulationRateMonitor(input_groups[name+'e'])    # rate_monitors['Xe'] = b2.PopulationRateMonitor(input_groups['Xe'])

for name in input_connection_names:                                               # input_connection_names = 'XA'
    print ('create connections between', name[0], 'and', name[1])                 # name[0] = 'X'  name[1] = 'A'
    for connType in input_conn_names:                                             # input_conn_names = ['ee_input']
        connName = name[0] + connType[0] + name[1] + connType[1]                  # connName = 'XeAe'
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')   # 取 XeAe.npy
        model = 'w : 1'
        pre = 'g%s_post += w' % connType[0]
        post = ''
        if ee_STDP_on:  # when it is on trianing, ee_STDP_on = True
            print ('create STDP for connection', name[0]+'e'+name[1]+'e')
            model += eqs_stdp_ee
            pre += '; ' + eqs_stdp_pre_ee
            post = eqs_stdp_post_ee

        connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                    model=model, on_pre=pre, on_post=post)               # connections['XeAe'] = b2.Synapses()

        # delay['ee_input'] = (0*b2.ms,10*b2.ms)
        # delay['ei_input'] = (0*b2.ms,5*b2.ms)
        
        minDelay = delay[connType][0]       # minDelay = 0 second
        maxDelay = delay[connType][1]       # maxDelay = 10 msecond
        deltaDelay = maxDelay - minDelay    # deltaDelay = 10 msecond
        # TODO: test this
        connections[connName].connect(True) # all-to-all connection                                # connections['XeAe'].connect(True)
        connections[connName].delay = 'minDelay + rand() * deltaDelay'                             # rand()会返回0~1的一个随机数，所以connections['XeAe'].delay是一个minDelay ~ maxDelay的一个随机数
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]   # 从weightMatrix这个矩阵中取对应的权重


#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

net = Network()
for obj_list in [neuron_groups, input_groups, connections, rate_monitors,
        spike_monitors, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

previous_spike_count = np.zeros(n_e)          # n_e = 400
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples            # num_examples = 30000 * 3
outputNumbers = np.zeros((num_examples, 10))  # (30000*3, 10)
input_weight_monitor, fig_weights = plot_2d_input_weights()  # return im2, fig   im2是画的内容， fig是画布
fig_num += 1
if do_plot_performance:  # when training, do_plot_performance = True
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)   # return im2, performance, fig_num, fig
for i,name in enumerate(input_population_names):   # input_population_names = ['X']
    input_groups[name+'e'].rates = 0 * Hz          # input_groups['Xe'].rates = 0 * Hz
net.run(0*second)
j = 0





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
from _globals import *
from pre_params import *
import os
from scipy import interpolate


#--------------------------------------------------
# 以下是一些 pre methods
#--------------------------------------------------

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print ('readout shape is:', readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


    # 获取二维输入权重
def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))   # (280, 400)

    connMatrix = np.zeros((n_input, n_e))                              # (280, 400)
    connMatrix[connections[name].i, connections[name].j] = connections[name].w       # 把输入矩阵权重的值又存入connMatrix中
    weight_matrix = np.copy(connMatrix)                                              # 对connMatrix进行了复制 

    return weight_matrix   # (280, 400)


def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b2.figure(fig_num, figsize = (18, 18))    # fig_num = 1
    im2 = b2.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b2.colorbar(im2)
    b2.title('weights of connection' + name)
    b2.show()
    return im2, fig


#---------------------------------------------
# 以下是LIF神经元模型和STDP突触模型
#---------------------------------------------


# 构建兴奋层和抑制层神经元组

record_spikes = True

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
        weightMatrix = get_matrix_from_file(weight_path + '/random/' + connName + ending + '.npy')
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
        weightMatrix = get_matrix_from_file(weight_path + '/random/' + connName + ending + '.npy')   # 取 XeAe.npy
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


# 上述代码已经将'XeAe', 'AeAi', 'AiAe' 的神经元模型和突触连接都已经完成, 且突触权重都已经初始化完成
#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

# net = Network()
# for obj_list in [neuron_groups, input_groups, connections, rate_monitors,
#         spike_monitors, spike_counters]:
#     for key in obj_list:
#         net.add(obj_list[key])

# previous_spike_count = np.zeros(n_e)          # n_e = 400
# assignments = np.zeros(n_e)
# input_numbers = [0] * num_examples            # num_examples = 30000 * 3
# outputNumbers = np.zeros((num_examples, 10))  # (30000*3, 10)
# input_weight_monitor, fig_weights = plot_2d_input_weights()  # return im2, fig   im2是画的内容， fig是画布
# fig_num += 1
# if do_plot_performance:  # when training, do_plot_performance = True
#     performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)   # return im2, performance, fig_num, fig
# for i,name in enumerate(input_population_names):   # input_population_names = ['X']
#     input_groups[name+'e'].rates = 0 * Hz          # input_groups['Xe'].rates = 0 * Hz
# net.run(0*second)
# j = 0

def plot_contour(Coordx,Coordy,Strain,figName,minX=0,maxX=500,minY=0,maxY=500):

    X = np.linspace(minX, maxX, 1000)
    Y = np.linspace(minY, maxY, 1000)
    #生成二维数据坐标点，可以想象成围棋棋盘上的一个个落子点
    X1, Y1 = np.meshgrid(X, Y)
    #通过griddata函数插值得到所有的(X1, Y1)处对应的值，原始数据为Coordx, Coordy, Strain
    Z = interpolate.griddata((Coordx, Coordy), Strain, (X1, Y1), method='cubic')
  
    fig, ax = plt.subplots(figName, figsize=(12, 8))

    #level设置云图颜色范围以及颜色梯度划分，只能显示0-601范围数值，每间隔20颜色变化
    levels = range(0,601,20) 
    cset1 = ax.contourf(X1, Y1, Z, levels,cmap=cm.jet) 
    #设置cmap为jet，最小值为蓝色，最大为红色，和有限元软件云图配色类似
   
    #设置图片在屏幕中出现的位置
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+350+50")

    ax.set_title( figName,size=20)  #设置图名
    #设置云图坐标范围以及坐标轴标签
    ax.set_xlim(minX, maxX)
    ax.set_ylim(minY, maxY)
    ax.set_xlabel("X(mm)",size=15)
    ax.set_ylabel("Y(mm)",size=15)

    #设置colorbar的刻度，标签
    cbar = fig.colorbar(cset1)
    cbar.set_label('strain(με)', size=18)
    cbar.set_ticks([0,100,200,300,400,500,600])
   
   
    #保存图片，bbox_inches设置图片周边空白的大小
    #fig.savefig(figName+".png", bbox_inches='tight',dpi=150,pad_inches=0.1)
    plt.show()  #是否显示图片
 
    return()




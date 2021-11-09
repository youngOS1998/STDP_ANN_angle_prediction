import torch
from pre_params import *
# from Spiking_model import *

#-------------------------------------------------------------------------
# start to supervised learning
#-------------------------------------------------------------------------
# 上面的 current_spike_count 和突出之间的 weight 都可以作为全连接网络层的输入

# 提取之前在 SNN 层中存好的 weights

# weight_XeAe = np.load(weight_save_path + '\XeAesave.npy')
# weight_AeAi = np.load(weight_save_path + '\AeAisave.npy')
# weight_AiAe = np.load(weight_save_path + '\AiAesave.npy')

# #--------------------------------

# #------------------------------------------------------------------------------
# # load the connection weights for 'AeAi' and 'AiAe'                    'AeAi'
# #------------------------------------------------------------------------------

# for subgroup_n, name in enumerate(population_names): # population_names = ['A']
#     print ('create neuron group', name)
#     for conn_type in recurrent_conn_names:  # recurrent_conn_names = ['ei', 'ie']
#         connName = name+conn_type[0]+name+conn_type[1] # 'AeAi' 'AiAe'
#         if connName == 'AeAi':
#             weightMatrix = create_Weight_Matrix(weight_AeAi, 400, 400)
#         else:
#             weightMatrix = create_Weight_Matrix(weight_AiAe, 400, 400)
#         connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]     # 从事先存好的矩阵数据中读取权重信息

# #------------------------------------------------------------------------------
# # load the connenction weights for 'XeAe'                              'XeAe'
# #------------------------------------------------------------------------------

# pop_values = [0,0,0]

# for name in input_connection_names:                                               # input_connection_names = 'XA'
#     print ('create connections between', name[0], 'and', name[1])                 # name[0] = 'X'  name[1] = 'A'
#     for connType in input_conn_names:                                             # input_conn_names = ['ee_input']
#         connName = name[0] + connType[0] + name[1] + connType[1]                  # connName = 'XeAe'
#         weightMatrix = create_Weight_Matrix(weight_XeAe, 280, 400)   # 取 XeAe.npy
#         connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]   # 从weightMatrix这个矩阵中取对应的权重

# #------------------------------------------------------------------------------

model = MyModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), 0.1)

n_epochs = 3000

current_data_x = np.random.rand(3000, 400)
current_data_y = np.random.rand(3000)

for epoch in range(n_epochs):
    model.train()
    # spike_rates = training['template'][j,:].reshape((n_input)) / 100. *  input_intensity
    # input_groups['Xe'].rates = spike_rates * Hz
    # net.run(single_example_time, report='text')

    # current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    # previous_spike_count = np.copy(spike_counters['Ae'].count[:])

    # data_x = current_spike_count[np.newaxis, :].to(device)
    # data_y = training['angle'][j].to(device)

    data_x = torch.tensor(current_data_x[epoch,:]).to(torch.float32).to(device)
    data_y = torch.tensor(current_data_y[epoch]).to(torch.float32).to(device)

    pred = model(data_x)
    loss = criterion(pred, data_y)
    loss.backward()
    print('loss is: %d', loss)
    optimizer.step()
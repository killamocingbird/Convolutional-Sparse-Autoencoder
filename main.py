import model
import torch
import torch.optim as optim
import os
import input_processing
import pickle
import math

#Import model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

do_train = True
do_infer = True
cont_train = True
iteration = 2
c = 8
sparsity_lambda = 1e-7

if do_train and not cont_train:
    net = model.SCAE(device=device)
else:
    net = torch.load('../../intermDat/SCAE' + str(iteration) + '/net', map_location=device)
    if do_train:
        optimizer = optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)
        optimizer.load_state_dict(torch.load('../../intermDat/SCAE' + str(iteration) + '/optim', map_location=device))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)    

if do_train:  
    #Training
    epochs = 500
    dataset = input_processing.raw_spectro_loader('../../Data/TIMIT')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    if not cont_train:
        optimizer = optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)
    out_freq = 1000
    print("-----Begining Training-----")
    min_loss = 1_000_000_000
    for epoch in range(epochs):
        running_loss = 0.0
        running_mse_loss = 0.0
        running_reg_loss = 0.0
        for i, data in enumerate(data_loader):
            inputs = data[-1,:,:]
            
            if i > 1001:
                continue
            
            #For now skip the ones that don't fit temporally
            if inputs.shape[0] > inputs.shape[1]:
                continue
            inputs = input_processing.dynamic_one_to_one_process(inputs, 10, net.encoding_layers)
            inputs = torch.reshape(inputs, [1, 1, inputs.shape[0], inputs.shape[1]])
            #inputs = data
            inputs = inputs.to(device)
            
            encoded, decoded = net(inputs)
            MSE_loss = (inputs - decoded)**2
            MSE_loss = c * 1000 * MSE_loss.mean()
#            MSE_loss = MSE_loss.view(1, -1).sum(1)
    
            regularization_loss = 0
            for activation_map in encoded:
                regularization_loss += torch.sum(torch.abs(activation_map))
            loss = MSE_loss + sparsity_lambda * regularization_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            running_mse_loss += MSE_loss
            running_reg_loss += sparsity_lambda * regularization_loss
            if (i + 1) % out_freq == 0:
                latest_loss = running_loss / out_freq / c
                print("[%d, %5d] loss: %.5f" % (epoch + 1, i + 1, latest_loss))
                print("M: %.5f \t R: %.5f \t RR: %.5f" % (running_mse_loss / out_freq / c, running_reg_loss / out_freq, running_reg_loss / out_freq / sparsity_lambda))
                if latest_loss < min_loss:
                    torch.save(net, '../../intermDat/SCAE' + str(iteration) + '/net')
                    torch.save(optimizer.state_dict(), '../../intermDat/SCAE' + str(iteration) + '/optim')
                    min_loss = latest_loss
                running_mse_loss = 0
                running_reg_loss = 0
                running_loss = 0
                

if do_infer:
    #Inferring
    dataset = input_processing.raw_spectro_loader('../../Data/TIMIT')
    out_dir = '../../intermDat/SCAE' + str(iteration)
    folder_names = ['Conv', 'Pool']
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    out_freq = 100
    print("-----Beginning Inferring-----")
    for i, data in enumerate(data_loader):
        inputs = data[-1,:,:]
        if inputs.shape[0] > inputs.shape[1]:
            continue
        inputs = input_processing.dynamic_one_to_one_process(inputs, 10, net.encoding_layers)
        inputs = torch.reshape(inputs, [1, 1, inputs.shape[0], inputs.shape[1]])
        inputs = inputs.to(device)
        encoded_maps = net.infer(inputs)
        for j in range(len(encoded_maps)):
            layer_idx = math.floor(j / 2) + 1
            folder_name = folder_names[j%2]
            with open(os.path.join(out_dir, folder_name + str(layer_idx), str(i + 1) + '.pickle'), 'wb') as out_file:
                pickle.dump(encoded_maps[j], out_file)
        if (i + 1)%out_freq == 0:
            print("%.2f%%" % (100 * (i + 1) / 4620))
            
        
        
    
    
    
    
    
    
    

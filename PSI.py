import os
import math
import pickle
import torch
import numpy as np
import tictoc as t
import F_ratio
from scipy.stats import ranksums
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

resume = False
resume_PSIs = False
response_calculations = [0,0,0,1,0,0,0,0,0,0,0,0]
PSI_calculations = [0,0,0,1,0,0,0,0,0,0,0,0]
out_channels = [100,100,200,200,500,500,500,500]
iteration = 1
use_sub = True
sub_num = 500
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#create dictionary for phoneme activation distribution
if resume:
    phoneme_distribs = pickle.load(open('../../intermDat/SCAE' + str(iteration) + '/phonemeDistribs.pickle', 'rb'))
else:
    phoneme_distribs = dict()
    #Extract Phoneme Map
    phoneme_Map = '../../Data/TIMIT/Phoneme Mapping.txt'
    print("Extracting phoneme map")
    with open(phoneme_Map, 'r') as pM:
        for line in pM:
            phoneme = line.split()[1]
            phoneme_distribs[phoneme] = [[[] for i in range(x)] for x in out_channels]

phoneme_dir = '../../Data/TIMIT'
folder_prefixes = ['../../intermDat/SCAE' + str(iteration) + '/Conv', '../../intermDat/SCAE' + str(iteration) + '/Pool']
tic = t.tic()
print("Calculating responses")
for i in range(len(out_channels)):
    tic()
    if not response_calculations[i]:
        continue
    print("Layer %s:" % (folder_prefixes[i%2][-4:] + str(math.floor(i / 2) + 1)))
    act_map_dir = folder_prefixes[i%2] + str(math.floor(i / 2) + 1)
    act_maps = os.listdir(act_map_dir)
    for file_idx in range(len(act_maps)):
        
        if use_sub and file_idx > sub_num:
            break
        
        #Load in map
        act_map = pickle.load(open(os.path.join(act_map_dir, act_maps[file_idx]), 'rb'))
        act_map.to(device)
        
        #Load in phoneme duration data
        phoneme_dat = pickle.load(open(os.path.join(phoneme_dir, act_maps[file_idx][:-7], 'procPhonemes.pickle'), 'rb'))
        
        for phoneme in phoneme_dat:
            if phoneme[0] in phoneme_distribs:
                phoneme_dur = phoneme[1][i + 2]
                for j in range(out_channels[i]):
                    phoneme_distribs[phoneme[0]][i][j].append(float(torch.max(torch.abs(act_map[0, j, :, phoneme_dur[0]:phoneme_dur[1] + 1]))))
                    
        if file_idx % 10 == 9:
            if use_sub:
               print("%.2f%% completed" % (100 * file_idx / sub_num))
            else:
                print("%.2f%% completed" % (100 * file_idx / len(act_maps)))
    tic.toc(setting='mins')

#Save distribs to intermDat
pickle.dump(phoneme_distribs, open('../../intermDat/SCAE' + str(iteration) + '/phonemeDistribs.pickle', 'wb'))

print("Determining active units")
active_units = [[] for x in range(len(out_channels))]
for i in range(len(out_channels)):
    if not PSI_calculations[i]:
        continue
    print("Map %d:" % (i + 1))
    tic()
    for j in range(out_channels[i]):
        not_silence = []
        for phoneme in phoneme_distribs:
            if phoneme != 'h#':
                not_silence.extend(phoneme_distribs[phoneme][i][j])
        rtest = ranksums(not_silence, phoneme_distribs['h#'][i][j])
        if rtest[0] > 0 and rtest[1] < 0.001:
            active_units[i].append(j)
    tic.toc()

    
    
print("Calculating PSI vectors")
if resume_PSIs:
    PSIs = pickle.load(open('../../intermDat/SCAE' + str(iteration) + '/PSIs.pickle', 'rb'))
    for i in range(len(PSI_calculations)):
        if PSI_calculations[i]:
            PSIs[i] = [[] for i in range(len(active_units[i]))]
        
else:
    PSIs = [[[] for i in range(len(x))] for x in active_units]
    
for i in range(len(out_channels)):
    if not PSI_calculations[i]:
        continue
    print("Map %d:" % (i + 1))
    tic()
    for j in range(len(active_units[i])):
        for x in phoneme_distribs:
            if x == 'h#':
                continue
            temp = 0
            for y in phoneme_distribs:
                if x != y and y != 'h#':
                    rtest = ranksums(phoneme_distribs[x][i][active_units[i][j]], phoneme_distribs[y][i][active_units[i][j]])
                    if rtest[0] > 0 and rtest[1] < 0.01:
                        temp += 1
            PSIs[i][j].append(temp)
        if j%10 == 9:
            print("%.2f%% completed" % (100 * (j + 1) / len(active_units[i])))
    tic.toc()

#Save PSIs to intermDat
pickle.dump(PSIs, open('../../intermDat/SCAE' + str(iteration) + '/PSIs.pickle', 'wb'))

#F-ratio
f_ratios = []

#Display PSI vectors for layer
for i in range(len(out_channels)):
    if not PSI_calculations[i]:
        continue
    map_name = folder_prefixes[i%2][-4:] + str(math.floor(i / 2) + 1)
    cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
    cluster.fit_predict(PSIs[i])
    temp = list(zip(cluster.labels_, PSIs[i]))
    temp.sort()
    f_ratios.append(F_ratio.calc(temp))
    temp2 = []
    for j in range(len(temp)):
        temp2.append(temp[j][1])
    
    PSIImg = np.array(temp2)
    PSIImg = PSIImg.transpose()
    
    plt.imshow(PSIImg, cmap='Greys')
    plt.savefig('../../outFigures/SCAE' + str(iteration) + '/PSIPlots_' + map_name + '.png')

print(*f_ratios)

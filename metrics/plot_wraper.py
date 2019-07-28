# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2018-09-27 23:41:05
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   BlueDreamStar
# @Last Modified time: 2019-07-28 14:48:21
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
from pylab import *
import matplotlib.pyplot as plt

## define the font
font = {'family' : 'serif',  
        'color'  : 'black',  
        'weight' : 'normal',  
        'size'   : 16,  
        }
## wraper the plot function
##############################
## Params:
## pType is the figure type, such as plot, bar
## iters is the painting interval along x-axis, in general, iters is generated by numpy.array
## data is the input data which will be draw in the figure
## colors is the color array for lines or bars
## axis_label is the label along x-axis & y-axis
## outSaveFile is the saveObjectFile
## lengendLoc is the location where lengend located
## lwValue is the line wieght of plot
## multiNum
## xlimit
## ylimit
## xtick
## ytick
## tickStep is the interval between two ticks in x-axis
##############################
def plot_save_figure(pType, iters, data, colors, labels, axis_label, outSaveFile, lengendLoc="upper right", lwValue=1.6, multiNum=2, xlimit=None, ylimit=None, xtick=None, ytick=None, tickStep=1):
	if len(colors) <= 0: return
# 	figure(1)
	foo_fig = plt.gcf() # get current figure
	
	if pType == 'bar':   
		total_width, n = 0.8, len(labels)
		width = total_width / n
		x = iters - (total_width - width) / 2
		
		if len(labels) == 1:
			plt.bar(x + 0 * width , data, facecolor = 'lightskyblue', width=width, label=labels[0])
		else:
			for i in range(len(labels)):
				plt.bar(x + i * width , data[i], lw=lwValue, width=width, label=labels[i])
	else:
		if len(labels) == 1:
			plt.plot(iters, data, colors[0], lw=lwValue, label=labels[0])
		else:
			for i in range(len(labels)):
				plt.plot(iters, data[i], colors[i], lw=lwValue, label=labels[i])
                
	plt.draw();
	ax=plt.gca()
	ax.set_xlabel(axis_label[0],fontdict=font)
	ax.set_ylabel(axis_label[1],fontdict=font)
	if not xlimit == None: plt.xlim((xlimit[0],xlimit[1]))
	if not ylimit == None: plt.ylim((ylimit[0],ylimit[1]))
	if not xtick == None: plt.xticks(iters[::tickStep], xtick)
	if not xtick == None: ax.set_xticklabels(xtick,rotation=90)
	if not ytick == None: plt.yticks(iters, ytick)
	xmajorLocator = MultipleLocator(multiNum)
	plt.legend(loc=lengendLoc, prop={'size': 16}) 
	plt.grid()
	foo_fig.savefig('%s.png' %(outSaveFile), bbox_inches='tight', format='png', dpi=1000)
	clf()
    
### return a list for ticks and a array for value
def read_label_value(fileInputDir):
	fileInput = open(fileInputDir)
	xtick = []
	dataValue = []
	for line in fileInput.readlines():
		lineArr = line.strip().split('\t')
		xtick.append(lineArr[0])
		dataValue.append(lineArr[1])
	dataValue = np.array(dataValue)
	return xtick, dataValue




# import matplotlib.pyplot as plt
# import torch
# from torch import nn
# from torch.autograd import Variable
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
# import torch.utils.data as Data
# import numpy as np
# import torch.nn.functional as F
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
# import os
# from pylab import *
# import matplotlib.pyplot as plt

# fileDir = "/Users/ppvsgg/Dropbox/SDM/Code/Outcome"
# labels=["Percent@50","Percent@60","Percent@70","Percent@80","Percent@90"]
# # labels=["Percent@50","Percent@60","Percent@70","Percent@80","Percent@90"]
# labelsX = ['20', '40', '60', '80', '100']
# # # plt.bar(range(len(data)), data, ec='k', lw=1, hatch='o')
# size = 5
# outComeLoss51 = np.array([68.2458, 60.0176, 53.874, 49.7999, 45.9424])
# outComeLoss61 = np.array([69.2049, 59.3654, 53.6226, 49.2424, 45.3888])
# outComeLoss71 = np.array([67.2432, 60.326, 53.6233, 49.7174, 45.387])
# outComeLoss81 = np.array([68.4418, 58.711, 54.362, 49.7235, 45.732])
# outComeLoss91 = np.array([70.9397, 60.4451, 54.7107, 48.539, 45.0869])

# outComeLoss62 = np.array([91.9317, 77.1518, 72.0968, 64.7815, 63.0124])
# outComeLoss63 = np.array([102.486, 88.38, 83.2415, 74.9359, 70.6591])
# outComeLoss64 = np.array([122.074, 102.289, 93.5097, 88.8865, 84.1524])
# outComeLoss65 = np.array([118.642, 101.696, 92.0568, 85.5868, 81.8041])
# x = np.arange(size)
# total_width, n = 0.8, 5
# width = total_width / n
# x = x - (total_width - width) / 2

# font = {'family' : 'serif',  
#         'color'  : 'black',  
#         'weight' : 'normal',  
#         'size'   : 16,  
#         } 
# figure()
# foo_fig = plt.gcf()
# plt.figure()
# plt.bar(x + 0 * width , outComeLoss51,  width=width, label=labels[0])
# plt.bar(x + 1 * width, outComeLoss61, width=width, label=labels[1])
# plt.bar(x + 2 * width, outComeLoss71, width=width, label=labels[2])
# plt.bar(x + 3 * width, outComeLoss81, width=width, label=labels[3])
# plt.bar(x + 4 * width, outComeLoss91, width=width, label=labels[4])
# # , tick_label=labelsX) 
# plt.draw(); 
# plt.xticks(np.arange(size), labelsX)
# # ax1 = plt.axes()
# # ax1.set_xticks(labelsX)'
# ax=plt.gca() 
# ax.set_ylabel('Cross Entroy',fontdict=font)  
# ax.set_xlabel('Batch Number',fontdict=font)
# xmajorLocator = MultipleLocator(2)
# plt.legend()
# # plt.show()
# foo_fig.savefig('%s/outComeLoss_2.pdf' %(fileDir), bbox_inches='tight', format='pdf', dpi=1000)
# # clf()
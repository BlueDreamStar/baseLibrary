# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2018-09-27 23:41:05
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   BlueDreamStar
# @Last Modified time: 2018-11-01 11:31:30
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
def plotAndSaveFigure(iters, data, colors, labels, axis_label, outSaveFile, lengendLoc="upper right", lwValue=1.6, multiNum=2, xlimit=None, ylimit=None, xtick=None, ytick=None):
	if len(colors) <= 0: return
# 	figure(1)
	foo_fig = plt.gcf() # get current figure
	if len(colors) == 1:
		plt.plot(iters, data, colors[0], lw=lwValue, label=labels[0])
	else:
		for i in range(len(colors)):
			plt.plot(iters, data[i], colors[i], lw=lwValue, label=labels[i])
	plt.draw();
	ax=plt.gca()
	ax.set_xlabel(axis_label[0],fontdict=font)
	ax.set_ylabel(axis_label[1],fontdict=font)
	if not xlimit == None: plt.xlim((xlimit[0],xlimit[1]))
	if not ylimit == None: plt.ylim((ylimit[0],ylimit[1]))
	if not xtick == None: plt.xticks(iters, xtick)
	if not xtick == None: ax.set_xticklabels(xtick,rotation=90)
	if not ytick == None: plt.yticks(iters, ytick)
	xmajorLocator = MultipleLocator(multiNum)
	plt.legend(loc=lengendLoc, prop={'size': 16}) 
	foo_fig.savefig('%s.pdf' %(outSaveFile), bbox_inches='tight', format='pdf', dpi=1000)
	clf()

# Copyright 2018 BlueDreamStar. All rights reserved.
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE file.
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
def plotAndSaveFigure(iters, data, colors, labels, axis_label, outSaveFile, lengendLoc="upper right", lwValue=1.6, multiNum=2, xlimit=None, ylimit=None):
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
	xmajorLocator = MultipleLocator(multiNum)
	plt.legend(loc=lengendLoc, prop={'size': 16}) 
	foo_fig.savefig('%s.pdf' %(outSaveFile), bbox_inches='tight', format='pdf', dpi=1000)
	clf()

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:54:15 2017

@author: oriol
"""

import math
import scipy.special as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def ClustCoeff(first_moment, second_moment, NN):
    return (1/NN)*np.power(second_moment - first_moment*first_moment, 2)/np.power(first_moment,3)

def SecondMoment(kave_list):
    if type(kave_list) == int:
        secondmoment = kave_list + kave_list*kave_list
    else:
        secondmoment = [(kave + kave*kave) for kave in kave_list]
    return secondmoment


###############################################################################
    
NN_list = [1000000] #for 'RW_Single'
NN_list = [10000] #for 'RW_Single'
kave_list = np.linspace(4, 25, num = 20)
nr = 10

###############################################################################

red  = '#dd181f'
blue = '#0060ad'
oran = '#e57300'
yellow = '#CC79A7'
green = '#228B22'
black = '#000000'
darkyellow = '#CCCC00'
gray = '#d3d3d3'
purple = '#6C006C'
colors = [blue, red, oran, yellow, green, black, darkyellow, gray, purple]


sizel = 25
lw = 3
ms = 14
sizelegend = 20
sizetext = 20
alpha = 0.1
plt.rcParams['xtick.major.pad'] = '10'
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': sizetext})

###############################################################################
###############################################################################
###############################################################################
## LOADING DATA ##



for NN in NN_list[:]:
    firstmoment = []
    secmoment = []

    for kave in kave_list:
        fname = '../Results_Asymptotic/N_'+str(NN)+'_kave_'+str('{:.6f}'.format(kave))+'_nrun_'+str(nr)
        fm = []
        sm = []
        for line in open(fname):
            if line[0] != '#':
                columns = line.split(' ')
                fm.append(float(columns[0]))
                sm.append(float(columns[1]))
        firstmoment.append(np.mean(fm))
        secmoment.append(np.mean(sm))

var_simus = [secmoment[i] - firstmoment[i] * firstmoment[i] for i in range(0, len(firstmoment))]

tl_coeff = 1
tl_exponent = 1
var_theor = [tl_coeff * firstmoment[i] ** tl_exponent for i in range(0, len(firstmoment))]

###############################################################################
###############################################################################
## Plot Moments

fig = plt.figure(1)
ax = fig.add_subplot(111)

plt.loglog(firstmoment, var_simus, marker = 'o', markerfacecolor='None',
            markeredgecolor='blue', markersize = ms,
            linestyle='None')

plt.loglog(firstmoment, var_theor, lw = lw, label = r'$ a \langle k \rangle ^ b $')


#plt.axhline(y=kave, ls='dashed', linewidth=0.5, color='black')
#plt.axhline(y=kave2_theory, ls='dashed', linewidth=0.5, color='black')

plt.legend(frameon = False)

plt.xlabel(r"$\langle k \rangle$",{'fontsize': sizel})
plt.ylabel(r"$var(k)$",{'fontsize': sizel})
plt.tick_params(labelsize=sizel)

ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

#plt.title('ER N='+str(NN)+'; kave='+str(kave)+'; '+str(expl_stra))

ax.text(0.6,0.2, r'$(a,b) = (%g, %g)$' % (tl_coeff, tl_exponent), horizontalalignment='left', verticalalignment='center', size = sizetext, transform=ax.transAxes)
#ax.text(0.4,0.65,r'ER; $\langle k \rangle = %d$' % kave, horizontalalignment='left', verticalalignment='center', size = sizetext, transform=ax.transAxes)

plt.tight_layout()

plt.title(r'ER $N=$'+str(NN))

plt.savefig('ER_N_'+str(NN)+'_Asymptotics.png', bbox_inches='tight')
plt.savefig('ER_N_'+str(NN)+'_Asymptotics.pdf', bbox_inches='tight')




































# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:54:15 2017

@author: oriol
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def ClustCoeff(first_moment, second_moment, NN):
    return (1/NN)*np.power(second_moment - first_moment*first_moment, 2)/np.power(first_moment,3)

def SecondMoment(kave_list):
    if type(kave_list) == int:
        secondmoment = kave_list + kave_list*kave_list
    else:
        secondmoment = [(kave + kave*kave) for kave in kave_list]
    return secondmoment


def fname(NN, initial_pos, target_position, left_boundary, right_boundary, nruns):
    return SCRIPT_DIR.parent / ('Results/N_'+str(NN)+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns))

###############################################################################
    
NN_list = [100] #for 'RW_Single'

NN = 100
initial_pos = 0.70 #When multiplied by N / 2 it has to result in a integer
initial_pos_list = [-0.90, -0.70, -0.30, 0, 0.30, 0.70, 0.90]
target_position = 1.
left_boundary = -1
right_boundary = -left_boundary
nruns = 10000


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
plt.rc('text', usetex=False)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': sizetext})

###############################################################################
###############################################################################
###############################################################################
## LOADING DATA ##

fptimes = {}
end_point = {}

for initial_pos in initial_pos_list:

    fptimes[initial_pos] = []
    end_point[initial_pos] = []
    
    for line in open(fname(NN, initial_pos, target_position, left_boundary, right_boundary, nruns), "r"):
    
        columns = line.split(' ')
    
        tt = float(columns[0]) / NN
        fptimes[initial_pos].append(tt)
        end_point[initial_pos].append(float(columns[1]))
    

###############################################################################
###############################################################################
## Data Processing

end_point_mean = {}
for key in end_point.keys():
    end_point_mean[key] = np.mean([1 if endpos == 1. else 0 for endpos in end_point[key] ])


xx_th = np.linspace(-1,1,30)
exit_prob_yy_th = 0.5 * (xx_th + 1)


fpt_mean = {}
for key in fptimes.keys():
    fpt_mean[key] = np.mean(fptimes[key])

fpt_xx_th = np.linspace(np.finfo(float).eps, 1 - np.finfo(float).eps, 30)
yy_th = -NN * ((1 - fpt_xx_th) * np.log(1 - fpt_xx_th) + fpt_xx_th * np.log(fpt_xx_th)) #This formula is for magns between 0 and 1

###############################################################################
###############################################################################
## Plot Moments

fig = plt.figure(2)
ax = fig.add_subplot(111)

plt.plot(xx_th, yy_th, marker = 'o', markersize = 0, lw = lw, label = r'Theory')
plt.plot(initial_pos_list, fpt_mean.values(), marker = 'o', markersize = 10, lw = 0, label = r'Sim')

plt.xlim((left_boundary, right_boundary))
# plt.ylim((0,1))

# plt.axhline(y=kave, ls='dashed', linewidth=0.5, color='black')
# plt.axhline(y=kave2_theory, ls='dashed', linewidth=0.5, color='black')

plt.legend(frameon = False)

plt.xlabel(r"$x_0$",{'fontsize': sizel})
plt.ylabel(r"$\langle T (x_0) \rangle$",{'fontsize': sizel})
plt.tick_params(labelsize=sizel)

plt.title('FPT Boundaries -- std voter')

#ax.text(0.4,0.8,r'\textsc{Triangles}', horizontalalignment='left', verticalalignment='center', size = sizetext, transform=ax.transAxes)
#ax.text(0.4,0.65,r'ER; $\langle k \rangle = %d$' % kave, horizontalalignment='left', verticalalignment='center', size = sizetext, transform=ax.transAxes)

plt.tight_layout()

# plt.title(r'ER $N=$'+str(NN)+' $\\langle k \\rangle =$ '+str(kave)+' -- '+expl_stra.replace('_', ' '))

plt.savefig(SCRIPT_DIR / 'VM_FPT_Mean.png', bbox_inches='tight')
# plt.savefig('ER_N_'+str(NN)+'_kave_'+str(kave)+'_'+str(expl_stra)+'_moments.pdf', bbox_inches='tight')

plt.show()







fig = plt.figure(1)
ax = fig.add_subplot(111)

plt.plot(xx_th, exit_prob_yy_th, marker = 'o', markersize = 0, lw = lw, label = r'Theory')
plt.plot(initial_pos_list, end_point_mean.values(), marker = 'o', markersize = 10, lw = 0, label = r'Sim')

plt.xlim((left_boundary, right_boundary))
plt.ylim((0,1))

# plt.axhline(y=kave, ls='dashed', linewidth=0.5, color='black')
# plt.axhline(y=kave2_theory, ls='dashed', linewidth=0.5, color='black')

plt.legend(frameon = False)

plt.xlabel(r"$x_0$",{'fontsize': sizel})
plt.ylabel(r"$E(x_0)$",{'fontsize': sizel})
plt.tick_params(labelsize=sizel)

plt.title('Exit prob -- std voter')

plt.tight_layout()

plt.savefig(SCRIPT_DIR / 'Exit_Prob_Std_VM.png', bbox_inches='tight')
# plt.savefig('ER_N_'+str(NN)+'_kave_'+str(kave)+'_'+str(expl_stra)+'_moments.pdf', bbox_inches='tight')

plt.show()

###############################################################################








# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:54:15 2017

@author: oriol
"""

import numpy as np
import matplotlib.pyplot as plt


def ClustCoeff(first_moment, second_moment, NN):
    return (1/NN)*np.power(second_moment - first_moment*first_moment, 2)/np.power(first_moment,3)

def SecondMoment(kave_list):
    if type(kave_list) == int:
        secondmoment = kave_list + kave_list*kave_list
    else:
        secondmoment = [(kave + kave*kave) for kave in kave_list]
    return secondmoment

def histo_fpt(fptimes, noise_p):
    xbins = np.linspace(min(fptimes[noise_p]), max(fptimes[noise_p]), num=50)
    xx = [0.5 * (xbins[i] + xbins[i+1]) for i in range(len(xbins) - 1)]
    histo, _ = np.histogram(fptimes[noise_p], bins = xbins, density = True)
    return xx, histo

def log_histo_fpt(fptimes, noise_p):
    xbins = np.logspace(min(np.log10(fptimes[noise_p])), max(np.log10(fptimes[noise_p])), num=20)
    xx = [0.5 * (xbins[i] + xbins[i+1]) for i in range(len(xbins) - 1)]
    histo, _ = np.histogram(fptimes[noise_p], bins = xbins, density = True)
    return xx, histo

def fname_Baseline(NN, noise_parm, initial_pos, target_position, left_boundary, right_boundary, nrun):
    return '../Results_Baseline/N_'+str(NN)+'_noise_'+format(noise_parm, '.5f')+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns)+'_Baseline'

def fname_Reshuffle(NN, noise_parm, initial_pos, target_position, left_boundary, right_boundary, nruns):
    return '../Results_Reshuffled/N_'+str(NN)+'_noise_'+format(noise_parm, '.5f')+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns)

###############################################################################
    
NN_list = [100] #for 'RW_Single'

NN = 100
noise_parm_list = [0.0001]
initial_pos = -1 #When multiplied by N / 2 it has to result in a integer
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
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': sizetext})

###############################################################################
###############################################################################
###############################################################################
## LOADING DATA ##

fptimes_bl = {}
end_point_bl = {}

fptimes_resh = {}
end_point_resh = {}

for noise_parm in noise_parm_list:

    fptimes_bl[noise_parm] = []
    end_point_bl[noise_parm] = []

    fptimes_resh[noise_parm] = []
    end_point_resh[noise_parm] = []
    
    for line in open(fname_Baseline(NN, noise_parm, initial_pos, target_position, left_boundary, right_boundary, nruns), "r"):
    
        columns = line.split(' ')
    
        tt = float(columns[0]) / NN
        fptimes_bl[noise_parm].append(tt)
        end_point_bl[noise_parm].append(float(columns[1]))

    for line in open(fname_Reshuffle(NN, noise_parm, initial_pos, target_position, left_boundary, right_boundary, nruns), "r"):
    
        columns = line.split(' ')
    
        tt = float(columns[0]) / NN
        fptimes_resh[noise_parm].append(tt)
        end_point_resh[noise_parm].append(float(columns[1]))
    

###############################################################################
###############################################################################
## Data Processing


xx_bl, histo_bl = histo_fpt(fptimes_bl, noise_parm) 
xx_resh, histo_resh = histo_fpt(fptimes_resh, noise_parm)

xx_bl, histo_bl = log_histo_fpt(fptimes_bl, noise_parm) 
xx_resh, histo_resh = log_histo_fpt(fptimes_resh, noise_parm)


end_point_bl_mean = {}
for key in end_point_bl.keys():
    end_point_bl_mean[key] = np.mean([1 if endpos == 1. else 0 for endpos in end_point_bl[key] ])


# xx_th = np.linspace(-1,1,30)
# exit_prob_yy_th = 0.5 * (xx_th + 1)


# fpt_bl_mean = {}
# for key in fptimes_bl.keys():
#     fpt_bl_mean[key] = np.mean(fptimes_bl[key])



###############################################################################
###############################################################################
## Plot Moments

fig = plt.figure(2)
ax = fig.add_subplot(111)

# plt.plot(xx_th, yy_th, marker = 'o', markersize = 0, lw = lw, label = r'Theory')
plt.loglog(xx_bl, histo_bl, marker = 'o', markersize = 10, lw = 0, label = r'BL')
plt.loglog(xx_resh, histo_resh, marker = 'o', markersize = 10, lw = 0, label = r'Resh')

# plt.loglog(xx_bl, np.power(xx_bl, -3/2))
# plt.loglog(xx_bl, np.power(xx_bl, -2))

# plt.xlim((left_boundary, right_boundary))
# plt.ylim((0,1))

# plt.axhline(y=kave, ls='dashed', linewidth=0.5, color='black')
# plt.axhline(y=kave2_theory, ls='dashed', linewidth=0.5, color='black')

plt.legend(frameon = False)

plt.xlabel(r"$t$",{'fontsize': sizel})
plt.ylabel(r"$f(1, t| -1)$",{'fontsize': sizel})
plt.tick_params(labelsize=sizel)

plt.title('FPT from -1 to 1 ; NVM noise '+format(noise_parm, '.5f'))

#ax.text(0.4,0.8,r'\textsc{Triangles}', horizontalalignment='left', verticalalignment='center', size = sizetext, transform=ax.transAxes)
#ax.text(0.4,0.65,r'ER; $\langle k \rangle = %d$' % kave, horizontalalignment='left', verticalalignment='center', size = sizetext, transform=ax.transAxes)

plt.tight_layout()

# plt.title(r'ER $N=$'+str(NN)+' $\\langle k \\rangle =$ '+str(kave)+' -- '+expl_stra.replace('_', ' '))

plt.savefig('N_'+str(NN)+'_noise_'+format(noise_parm, '.5f')+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns)+'.png', bbox_inches='tight')
# plt.savefig('ER_N_'+str(NN)+'_kave_'+str(kave)+'_'+str(expl_stra)+'_moments.pdf', bbox_inches='tight')

plt.show()







# fig = plt.figure(1)
# ax = fig.add_subplot(111)

# plt.plot(xx_th, exit_prob_yy_th, marker = 'o', markersize = 0, lw = lw, label = r'Theory')
# plt.plot(initial_pos_list, end_point_mean.values(), marker = 'o', markersize = 10, lw = 0, label = r'Sim')

# plt.xlim((left_boundary, right_boundary))
# plt.ylim((0,1))

# # plt.axhline(y=kave, ls='dashed', linewidth=0.5, color='black')
# # plt.axhline(y=kave2_theory, ls='dashed', linewidth=0.5, color='black')

# plt.legend(frameon = False)

# plt.xlabel(r"$x_0$",{'fontsize': sizel})
# plt.ylabel(r"$E(x_0)$",{'fontsize': sizel})
# plt.tick_params(labelsize=sizel)

# plt.title('Exit prob -- std voter')

# plt.tight_layout()

# plt.savefig('Exit_Prob_Std_VM.png', bbox_inches='tight')
# # plt.savefig('ER_N_'+str(NN)+'_kave_'+str(kave)+'_'+str(expl_stra)+'_moments.pdf', bbox_inches='tight')

# plt.show()

###############################################################################








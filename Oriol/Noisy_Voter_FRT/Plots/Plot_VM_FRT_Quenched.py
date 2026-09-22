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
    return '../Results_FRT_Baseline/N_'+str(NN)+'_noise_'+format(noise_parm, '.5f')+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns)+'_Baseline'

def fname_Quenched(NN, noise_parm, initial_pos, target_position, left_boundary, right_boundary, nruns):
    return '../Results_FRT_Quenched/N_'+str(NN)+'_noise_'+format(noise_parm, '.5f')+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns)

###############################################################################
    
NN_list = [100] #for 'RW_Single'

NN = 100
noise_parm_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
noise_parm_firstplot = 0.01
initial_pos = -0.98 #When multiplied by N / 2 it has to result in a integer
initial_pos_list = [-0.90, -0.70, -0.30, 0, 0.30, 0.70, 0.90]
target_position = -1.
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

    for line in open(fname_Quenched(NN, noise_parm, initial_pos, target_position, left_boundary, right_boundary, nruns), "r"):
    
        columns = line.split(' ')
        
        if len(columns) == 2:
            tt = float(columns[0]) / NN
            fptimes_resh[noise_parm].append(tt)
            end_point_resh[noise_parm].append(float(columns[1]))
    

###############################################################################
###############################################################################
## Data Processing

noise_parm = noise_parm_firstplot

xx_bl, histo_bl = histo_fpt(fptimes_bl, noise_parm) 
xx_resh, histo_resh = histo_fpt(fptimes_resh, noise_parm)

xx_bl, histo_bl = log_histo_fpt(fptimes_bl, noise_parm) 
xx_resh, histo_resh = log_histo_fpt(fptimes_resh, noise_parm)


end_point_bl_mean = {}
for key in end_point_bl.keys():
    end_point_bl_mean[key] = np.mean([1 if endpos == 1. else 0 for endpos in end_point_bl[key] ])



###############################################################################
###############################################################################
## Plot Moments

fig = plt.figure(2)
ax = fig.add_subplot(111)

plt.loglog(xx_bl, histo_bl, marker = 'o', markersize = 10, lw = 0, label = r'BL')
plt.loglog(xx_resh, histo_resh, marker = 'o', markersize = 10, lw = 0, label = r'Quenched')

if noise_parm == 0.0001:
    plt.loglog(xx_bl, 0.1*np.power(xx_bl, -3/2))
    ii = 5
    plt.loglog(xx_bl[ii:], np.power(xx_bl[ii:], -2))

if noise_parm == 0.001:
    plt.loglog(xx_bl, 0.1*np.power(xx_bl, -3/2))
    ii = 5
    plt.loglog(xx_bl[ii:], np.power(xx_bl[ii:], -2))
    
if noise_parm == 0.005:
    plt.loglog(xx_bl, 0.1*np.power(xx_bl, -3/2))
    ii = 5
    plt.loglog(xx_bl[ii:], 10*np.power(xx_bl[ii:], -2))

if noise_parm == 0.01:
    plt.loglog(xx_bl, 0.1*np.power(xx_bl, -3/2))
    ii = 5
    plt.loglog(xx_bl[ii:], 10*np.power(xx_bl[ii:], -2))


plt.legend(frameon = False)

plt.xlabel(r"$t$",{'fontsize': sizel})
plt.ylabel(r"$f(1, t| -1)$",{'fontsize': sizel})
plt.tick_params(labelsize=sizel)

plt.title('FPT from '+format(initial_pos, 'g')+' to '+format(target_position, 'g')+' ; NVM noise '+format(noise_parm, 'g'))

plt.tight_layout()

plt.savefig('Quenched_N_'+str(NN)+'_noise_'+format(noise_parm, '.5f')+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns)+'.png', bbox_inches='tight')

plt.show()




###############################################################################


fig = plt.figure(1)
ax = fig.add_subplot(111)


for noise_parm in noise_parm_list:
    
    xx_resh, histo_resh = histo_fpt(fptimes_resh, noise_parm) # or
    xx_resh, histo_resh = log_histo_fpt(fptimes_resh, noise_parm)
    

    plt.loglog(xx_resh, histo_resh, markersize = 7.5, marker = 'o', lw = 0, markeredgewidth = 2, markerfacecolor='None', label = r'$%g$' % noise_parm)

plt.loglog(xx_resh, 0.01*np.power(xx_resh, -3/2), lw = 2)
plt.loglog(xx_resh, 0.01*np.power(xx_resh, -2), lw = 2)

plt.legend(frameon = False,
            prop={'size': 12})


plt.xlabel(r"$t$",{'fontsize': sizel})
plt.ylabel(r"$f(%g, t| %g)$" % (target_position, initial_pos), {'fontsize': sizel})
plt.tick_params(labelsize=sizel)

plt.title('FPT from '+format(initial_pos, 'g')+' to '+format(target_position, 'g')+' (Quenched)')

plt.tight_layout()

plt.savefig('FRT_Quenched_vsNoise.png', bbox_inches='tight')

plt.show()

###############################################################################

fig = plt.figure(3)
ax = fig.add_subplot(111)


for noise_parm in noise_parm_list:
    xx_bl, histo_bl = histo_fpt(fptimes_bl, noise_parm) # or    
    xx_bl, histo_bl = log_histo_fpt(fptimes_bl, noise_parm)
    
    plt.loglog(xx_bl, histo_bl, markersize = 7.5, marker = 'o', lw = 0, markeredgewidth = 2, markerfacecolor='None', label = r'$%g$' % noise_parm)


plt.loglog(xx_resh, 0.01*np.power(xx_resh, -3/2), lw = 2)
ii = 5
plt.loglog(xx_resh[ii:], 10*np.power(xx_resh[ii:], -2), lw = 2)

plt.legend(frameon = False,
            prop={'size': 12})


plt.xlabel(r"$t$",{'fontsize': sizel})
plt.ylabel(r"$f(%g, t| %g)$" % (target_position, initial_pos), {'fontsize': sizel})
plt.tick_params(labelsize=sizel)

plt.title('FPT from '+format(initial_pos, 'g')+' to '+format(target_position, 'g'))

plt.tight_layout()

plt.savefig('FRT_Baseline_vsNoise.png', bbox_inches='tight')

plt.show()

###############################################################################







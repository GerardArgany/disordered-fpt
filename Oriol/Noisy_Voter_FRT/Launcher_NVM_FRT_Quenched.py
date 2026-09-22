import os
import numpy as np

def check_pos(pos, N):
    if pos * N / 2 - int(pos * N / 2):
        raise ValueError
    return

#noise_parm_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

NN = 100

noise_parm = 0.05
initial_pos = -0.98 #When multiplied by N / 2 it has to result in a integer
target_position = -1.
left_boundary = -1
right_boundary = -left_boundary
nruns = 10000

check_pos(initial_pos, NN)
check_pos(target_position, NN)

randseed = np.random.randint(0, high = 2147483647, size = None)

script_name = "NVM_FRT_Quenched.c"
# ~ script_name_out = "Teams_N%d.c" % NN
# ~ os.system("sed '1 i\#define NN %d' %s > %s" % (NN, script_name_in, script_name_out))

exe_name = "aa_%g_%g_%g_%g_%g_%g_%g.out" % (NN, initial_pos, target_position, left_boundary, right_boundary, nruns, randseed, )

os.system("gcc -g -o %s %s -lm" % (exe_name, script_name))
    
fname='input_N_'+str(NN)+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns)+'.in'
fout1='Results_FRT_Quenched/N_'+str(NN)+'_noise_'+format(noise_parm, '.5f')+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns)
    
finput = open(fname,'w+')
finput.write('%d ' % NN)
finput.write('%f ' % noise_parm)
finput.write('%f ' % initial_pos)
finput.write('%f ' % target_position)
finput.write('%f ' % left_boundary)
finput.write('%f ' % right_boundary)
finput.write('%d ' % nruns)
finput.write(fout1+' ')
finput.close()
    
os.system('./'+str(exe_name)+' '+fname+' '+str(randseed)) #run with valgrind when updates are made, to check memory is OK
    
# os.system('valgrind --leak-check=yes --track-origins=yes ./'+str(exe_name)+' '+fname+' '+str(randseed))
    
os.system("rm %s" % (exe_name))






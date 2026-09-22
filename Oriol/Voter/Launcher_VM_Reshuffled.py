import shutil
import subprocess
from pathlib import Path
import numpy as np

def check_pos(pos, N):
    if pos * N / 2 - int(pos * N / 2):
        raise ValueError
    return

NN = 100

initial_pos = 0.94 #When multiplied by N / 2 it has to result in a integer
target_position = 1.
left_boundary = -1
right_boundary = -left_boundary
nruns = 1

check_pos(initial_pos, NN)
check_pos(target_position, NN)

randseed = np.random.randint(0, high = 2147483647, size = None)

script_dir = Path(__file__).resolve().parent
script_name = script_dir / "VM_Reshuffled.c"
# ~ script_name_out = "Teams_N%d.c" % NN
# ~ os.system("sed '1 i\#define NN %d' %s > %s" % (NN, script_name_in, script_name_out))



exe_name = script_dir / ("aa_%g_%g_%g_%g_%g_%g_%g.exe" % (NN, initial_pos, target_position, left_boundary, right_boundary, nruns, randseed, ))

compiler = shutil.which("gcc")
if compiler is None:
    raise RuntimeError("GCC was not found on PATH. Install MinGW-w64 or GCC and retry.")
subprocess.run([compiler, "-g", "-o", str(exe_name), str(script_name), "-lm"], check=True)
    
fname = script_dir / ('input_N_'+str(NN)+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns)+'.in')
fout1 = script_dir / ('Results/N_'+str(NN)+'_init_'+str(initial_pos)+'_target_'+str(target_position)+'_left_'+str(left_boundary)+'_right_'+str(right_boundary)+'_nrun_'+str(nruns))
    
with open(fname, 'w+') as finput:
    finput.write('%d ' % NN)
    finput.write('%f ' % initial_pos)
    finput.write('%f ' % target_position)
    finput.write('%f ' % left_boundary)
    finput.write('%f ' % right_boundary)
    finput.write('%d ' % nruns)
    finput.write(str(fout1) + ' ')

try:
    subprocess.run([str(exe_name), str(fname), str(randseed)], check=True, cwd=script_dir)
finally:
    exe_name.unlink(missing_ok=True)
    fname.unlink(missing_ok=True)






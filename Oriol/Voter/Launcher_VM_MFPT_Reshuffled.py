import shutil
import subprocess
from pathlib import Path
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
N = 500
LEFT_BOUNDARY = -1.0
RIGHT_BOUNDARY = 1.0
SAMPLES_PER_SHUFFLE = 5000
RESHUFFLES = 5000
SEED = int(np.random.default_rng().integers(0, 2**31 - 1))

compiler = shutil.which('gcc')
if compiler is None:
    strawberry_gcc = Path(r'C:\Strawberry\c\bin\gcc.exe')
    if strawberry_gcc.exists():
        compiler = str(strawberry_gcc)
if compiler is None:
    raise RuntimeError('GCC was not found on PATH.')

executable = SCRIPT_DIR / f'vm_mfpt_reshuffled_{SEED}.exe'
input_file = SCRIPT_DIR / 'vm_mfpt_input.in'
output_file = SCRIPT_DIR / 'Results' / (
    f'MFPT_reshuffled_N_{N}_from_left_nearest_to_right_'
    f'{SAMPLES_PER_SHUFFLE}samples_{RESHUFFLES}reshuffles'
)

subprocess.run([
    compiler, '-O2', '-o', str(executable),
    str(SCRIPT_DIR / 'VM_MFPT_Reshuffled.c'), '-lm'
], check=True)

with input_file.open('w') as input_stream:
    input_stream.write(
        f'{N} {LEFT_BOUNDARY} {RIGHT_BOUNDARY} '
        f'{SAMPLES_PER_SHUFFLE} {RESHUFFLES} {output_file}\n'
    )

try:
    subprocess.run([str(executable), str(input_file), str(SEED)], check=True, cwd=SCRIPT_DIR)
finally:
    executable.unlink(missing_ok=True)
    input_file.unlink(missing_ok=True)
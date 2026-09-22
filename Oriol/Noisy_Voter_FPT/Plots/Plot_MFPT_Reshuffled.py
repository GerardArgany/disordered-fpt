import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / 'Results_Reshuffled'
N = 100
NOISE = 0.05
INITIAL_POSITION = -1
TARGET_POSITION = 1.0
NRUNS = 100000
BLOCK_SIZE = 1000


result_file = RESULTS_DIR / (
    'N_' + str(N) + '_noise_' + format(NOISE, '.5f')
    + '_init_' + str(INITIAL_POSITION)
    + '_target_' + str(TARGET_POSITION)
    + '_left_-1_right_1_nrun_' + str(NRUNS)
)

samples = []
with result_file.open() as result:
    for line in result:
        columns = line.split()
        if len(columns) >= 2:
            passage_time = float(columns[0])
            if passage_time >= 0:
                samples.append(passage_time / N)

fptimes = np.asarray(samples)
if fptimes.size < BLOCK_SIZE:
    raise ValueError(f'Need at least {BLOCK_SIZE} valid reshuffled runs, found {fptimes.size}.')

block_count = fptimes.size // BLOCK_SIZE
mfpt_samples = fptimes[:block_count * BLOCK_SIZE].reshape(block_count, BLOCK_SIZE).mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

positive_fptimes = fptimes[fptimes > 0]
log_bins = np.logspace(np.log10(positive_fptimes.min()), np.log10(positive_fptimes.max()), 60)
axes[0].hist(positive_fptimes, bins=log_bins, density=True, color='#e57300', alpha=0.78)
axes[0].axvline(fptimes.mean(), color='#0060ad', linewidth=2.5, label=f'Mean = {fptimes.mean():.2f}')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel('FPT $t/N$')
axes[0].set_ylabel('Density')
axes[0].set_title('FPT samples across reshufflings')
axes[0].legend(frameon=False)

axes[1].hist(mfpt_samples, bins=min(30, block_count), density=True, color='#dd181f', alpha=0.78)
axes[1].axvline(mfpt_samples.mean(), color='#0060ad', linewidth=2.5,
                 label=f'Mean = {mfpt_samples.mean():.2f}')
axes[1].set_xlabel(f'Block mean FPT ($\\mathrm{{{BLOCK_SIZE}}}$ reshufflings)')
axes[1].set_ylabel('Density')
axes[1].set_title('MFPT estimates from reshuffling blocks')
axes[1].legend(frameon=False)

fig.suptitle(f'Reshuffled noisy voter model ($N={N}$, noise={NOISE:g}, samples={fptimes.size})')
fig.tight_layout()
fig.savefig(SCRIPT_DIR / 'MFPT_Distribution_Reshuffled.png', dpi=160, bbox_inches='tight')
plt.show()
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / 'Results'
N = 500
SAMPLES_PER_SHUFFLE = 5000
RESHUFFLES = 5000

result_file = RESULTS_DIR / (
    f'MFPT_reshuffled_N_{N}_from_left_nearest_to_right_'
    f'{SAMPLES_PER_SHUFFLE}samples_{RESHUFFLES}reshuffles'
)
fpt_result_file = Path(str(result_file) + '_FPT_samples')

rows = []
with result_file.open() as result:
    for line in result:
        columns = line.split()
        if len(columns) == 3:
            rows.append([float(value) for value in columns])
if not rows:
    rows = []

fpt_rows = []
with fpt_result_file.open() as result:
    for line in result:
        columns = line.split()
        if len(columns) == 2:
            fpt_rows.append([float(value) for value in columns])
if not fpt_rows:
    raise ValueError(f'No complete FPT samples found in {fpt_result_file}.')
fpt_data = np.asarray(fpt_rows)
fptimes = fpt_data[:, 1]

if rows:
    data = np.asarray(rows)
    mfpt = data[:, 1]
    completed_samples = data[:, 2]
else:
    shuffle_ids = fpt_data[:, 0].astype(int)
    unique_shuffles = np.unique(shuffle_ids)
    mfpt = np.asarray([fptimes[shuffle_ids == shuffle_id].mean() for shuffle_id in unique_shuffles])
    completed_samples = np.asarray([np.sum(shuffle_ids == shuffle_id) for shuffle_id in unique_shuffles])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
positive_fptimes = fptimes[fptimes > 0]
log_bins = np.logspace(np.log10(positive_fptimes.min()), np.log10(positive_fptimes.max()), 60)
axes[0].hist(positive_fptimes, bins=log_bins, density=True, color='#e57300', alpha=0.8)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel(r'FPT $t/N$')
axes[0].set_ylabel('Density')
axes[0].set_title(f'FPT samples over {len(mfpt)} shuffled landscapes')

axes[1].hist(mfpt, bins=40, density=True, color='#0060ad', alpha=0.8)
axes[1].axvline(mfpt.mean(), color='#dd181f', linewidth=2.5,
                label=f'Mean = {mfpt.mean():.2f}')
axes[1].set_xlabel(r'MFPT $\langle T \rangle/N$')
axes[1].set_ylabel('Density')
axes[1].set_title(f'MFPT over {len(mfpt)} shuffled landscapes')
axes[1].legend(frameon=False)
fig.tight_layout()
output_figure = SCRIPT_DIR / 'FPT_and_MFPT_Distributions_5000_Reshuffles.png'
fig.savefig(output_figure, dpi=160, bbox_inches='tight')
plt.close(fig)

fpt_figure, fpt_axis = plt.subplots(figsize=(8, 5))
fpt_axis.hist(positive_fptimes, bins=log_bins, density=True, color='#e57300', alpha=0.8)
fpt_axis.set_xscale('log')
fpt_axis.set_yscale('log')
fpt_axis.set_xlabel(r'FPT $t/N$')
fpt_axis.set_ylabel('Density')
fpt_axis.set_title(f'Plain voter FPT samples over shuffled landscapes (n={len(fptimes)})')
fpt_figure.tight_layout()
fpt_figure.savefig(SCRIPT_DIR / 'FPT_Distribution_Over_Shuffles.png', dpi=160, bbox_inches='tight')
plt.close(fpt_figure)
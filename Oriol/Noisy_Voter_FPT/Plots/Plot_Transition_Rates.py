import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
N = 100
NOISE = 0.05
SEED = 12345


def noisy_rates(noise, positions):
    total_rate = 0.5 * (1 - noise) * (1 - positions * positions) + noise
    drift_rate = -noise * positions
    return total_rate, drift_rate


positions = np.linspace(-1, 1, N + 1)
total_rate, drift_rate = noisy_rates(NOISE, positions)
interior = np.arange(1, N)
rng = np.random.default_rng(SEED)

quenched_order = interior.copy()
rng.shuffle(quenched_order)
quenched_total = total_rate.copy()
quenched_drift = drift_rate.copy()
quenched_total[interior] = total_rate[quenched_order]
quenched_drift[interior] = drift_rate[quenched_order]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
for axis, baseline, quenched, ylabel in zip(
    axes,
    (total_rate, drift_rate),
    (quenched_total, quenched_drift),
    (r'$p_L+p_R$', r'$p_R-p_L$'),
):
    axis.plot(positions, baseline, color='#0060ad', linewidth=2.5, label='Baseline')
    axis.plot(positions, quenched, color='#dd181f', linewidth=1.2, label='Quenched')

    for _ in range(5):
        reshuffled_order = interior.copy()
        rng.shuffle(reshuffled_order)
        reshuffled = baseline.copy()
        reshuffled[interior] = baseline[reshuffled_order]
        axis.plot(positions, reshuffled, color='#e57300', alpha=0.28, linewidth=1)

    axis.set_xlabel(r'Position $x$')
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    axis.set_xlim(-1, 1)

axes[0].legend(frameon=False)
fig.suptitle(f'Noisy voter transition-rate orderings ($N={N}$, noise={NOISE:g})')
fig.tight_layout()
fig.savefig(SCRIPT_DIR / 'Transition_Rate_Orderings.png', dpi=160, bbox_inches='tight')
plt.show()
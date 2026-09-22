from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
N = 500
LEFT_BOUNDARY = -1.0
RIGHT_BOUNDARY = 1.0


def voter_rates(n):
    positions = np.linspace(LEFT_BOUNDARY, RIGHT_BOUNDARY, n + 1)
    rates = 0.25 * (1.0 - positions * positions)
    return positions[1:-1], rates[1:-1]


def mfpt_from_rates(rates):
    indices = np.arange(1, N)
    return float(np.sum((N - indices) / rates))


def pairwise_relaxed_condition(rates):
    half = len(rates) // 2
    return bool(np.all(rates[::-1][:half] + 1e-14 >= rates[:half]))


def make_orderings(base_rates):
    sorted_rates = np.sort(base_rates)
    midpoint = len(sorted_rates) // 2
    low = sorted_rates[:midpoint]
    high = sorted_rates[-midpoint:]
    middle = sorted_rates[midpoint:midpoint + 1]

    decrease_increase = np.concatenate((low[::-1], middle, high))
    decrease_decrease = np.concatenate((low[::-1], middle, high[::-1]))

    return {
        'Increasing': sorted_rates,
        'Decreasing': sorted_rates[::-1],
        'Decrease-increase\n(relaxed)': decrease_increase,
        'Reverse decrease-increase\n(increase-decrease)': decrease_increase[::-1],
        'Decrease-decrease\n(big middle jump, relaxed)': decrease_decrease,
        'Reverse decrease-decrease\n(decrease-increase jump)': decrease_decrease[::-1],
    }


positions, base_rates = voter_rates(N)
orderings = make_orderings(base_rates)
results = {}
for name, rates in orderings.items():
    results[name] = {
        'mfpt': mfpt_from_rates(rates),
        'pairwise_condition': pairwise_relaxed_condition(rates),
        'monotonic_increasing': bool(np.all(np.diff(rates) >= 0)),
        'monotonic_decreasing': bool(np.all(np.diff(rates) <= 0)),
    }

reference = min(item['mfpt'] for item in results.values())
for item in results.values():
    item['ratio_to_smallest'] = item['mfpt'] / reference

with (SCRIPT_DIR / 'Rate_Ordering_Comparison.txt').open('w') as report:
    report.write(f'N={N}\n')
    report.write('MFPT formula: sum((N-i)/b_i), i=1,...,N-1\n\n')
    for name, item in results.items():
        report.write(
            f"{name.replace(chr(10), ' ')}: MFPT={item['mfpt']:.12g}, "
            f"ratio={item['ratio_to_smallest']:.8g}, "
            f"pairwise_relaxed={item['pairwise_condition']}, "
            f"increasing={item['monotonic_increasing']}, "
            f"decreasing={item['monotonic_decreasing']}\n"
        )

names = list(orderings)
mfpt_values = [results[name]['mfpt'] for name in names]
colors = ['#0060ad', '#e57300', '#228B22', '#8c564b', '#dd181f', '#9467bd']

fig, profile_axes = plt.subplots(3, 2, figsize=(12, 13), sharex=True, sharey=True)
for axis, (name, rates), color in zip(profile_axes.flat, orderings.items(), colors):
    axis.plot(positions, rates, linewidth=2.2, color=color)
    axis.axvline(0, color='black', linewidth=0.8, alpha=0.35)
    axis.set_title(name.replace('\n', ' '), fontsize=10)
    axis.grid(alpha=0.2)
    axis.set_xlim(-1, 1)
    axis.set_ylim(0, 0.265)
for axis in profile_axes[-1, :]:
    axis.set_xlabel('Position $x$')
for axis in profile_axes[:, 0]:
    axis.set_ylabel('$b_i=d_i$')
fig.suptitle('Transition-rate orderings', fontsize=15)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / 'Rate_Ordering_Profiles.png', dpi=160, bbox_inches='tight')
plt.close(fig)

fig, axis = plt.subplots(figsize=(11, 5.5))
bars = axis.bar(
    np.arange(len(names)), mfpt_values,
    color=colors
)
axis.set_xticks(np.arange(len(names)), names, rotation=18, ha='right')
axis.set_ylabel('$T_1$')
axis.set_title('Exact MFPT from $x=-1+2/N$ to $x=1$')
axis.grid(axis='y', alpha=0.2)
for bar, value in zip(bars, mfpt_values):
    axis.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.0f}',
                 ha='center', va='bottom', fontsize=9)

fig.savefig(SCRIPT_DIR / 'Rate_Ordering_MFPT_Comparison.png', dpi=160, bbox_inches='tight')
plt.close(fig)

for name, item in results.items():
    print(f"{name.replace(chr(10), ' ')}: MFPT={item['mfpt']:.12g}, "
          f"pairwise_relaxed={item['pairwise_condition']}")
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / 'Results'
N = 500
SAMPLES_PER_SHUFFLE = 5000
RESHUFFLES = 5000

RESULT_FILE = RESULTS_DIR / (
    f'MFPT_reshuffled_N_{N}_from_left_nearest_to_right_'
    f'{SAMPLES_PER_SHUFFLE}samples_{RESHUFFLES}reshuffles'
)
FPT_RESULT_FILE = Path(str(RESULT_FILE) + '_FPT_samples')


def read_rows(path, width):
    rows = []
    with path.open() as stream:
        for line in stream:
            values = line.split()
            if len(values) == width:
                rows.append([float(value) for value in values])
    return np.asarray(rows)


def reconstruct_mfpt(fpt_data):
    shuffle_ids = fpt_data[:, 0].astype(int)
    unique_ids = np.unique(shuffle_ids)
    means = np.asarray([fpt_data[shuffle_ids == identifier, 1].mean() for identifier in unique_ids])
    counts = np.asarray([np.sum(shuffle_ids == identifier) for identifier in unique_ids])
    return means, counts


def fit_distribution(name, distribution, values, parameter_names):
    if name == 'normal':
        parameters = distribution.fit(values)
    else:
        parameters = distribution.fit(values, floc=0)
    log_likelihood = float(np.sum(distribution.logpdf(values, *parameters)))
    parameter_count = len(parameters)
    statistic, p_value = stats.kstest(
        values, lambda sample: distribution.cdf(sample, *parameters)
    )
    return {
        'distribution': name,
        'parameters': dict(zip(parameter_names, [float(value) for value in parameters])),
        'log_likelihood': log_likelihood,
        'AIC': 2 * parameter_count - 2 * log_likelihood,
        'BIC': parameter_count * np.log(len(values)) - 2 * log_likelihood,
        'KS': float(statistic),
        'KS_pvalue': float(p_value),
    }


def fit_power_law_tail(values):
    sorted_values = np.sort(values)
    candidates = np.unique(np.quantile(sorted_values, np.linspace(0.5, 0.95, 46)))
    best = None
    for xmin in candidates:
        tail = sorted_values[sorted_values >= xmin]
        if len(tail) < 50 or tail[-1] <= xmin:
            continue
        alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
        empirical_survival = np.arange(len(tail), 0, -1) / len(tail)
        model_survival = (tail / xmin) ** (1 - alpha)
        ks = float(np.max(np.abs(empirical_survival - model_survival)))
        candidate = {
            'distribution': 'power_law_tail',
            'xmin': float(xmin),
            'alpha': float(alpha),
            'tail_count': int(len(tail)),
            'tail_fraction': float(len(tail) / len(values)),
            'KS': ks,
        }
        if best is None or candidate['KS'] < best['KS']:
            best = candidate
    return best


def analyze(name, values):
    values = np.asarray(values)
    values = values[np.isfinite(values) & (values > 0)]
    models = [
        ('normal', stats.norm, ('loc', 'scale')),
        ('exponential', stats.expon, ('loc', 'scale')),
        ('gamma', stats.gamma, ('shape', 'loc', 'scale')),
        ('weibull', stats.weibull_min, ('shape', 'loc', 'scale')),
        ('lognormal', stats.lognorm, ('shape', 'loc', 'scale')),
    ]
    fits = [fit_distribution(model_name, distribution, values, parameters)
            for model_name, distribution, parameters in models]
    return {
        'sample_count': int(len(values)),
        'mean': float(values.mean()),
        'standard_deviation': float(values.std(ddof=1)),
        'median': float(np.median(values)),
        'fits': sorted(fits, key=lambda fit: fit['AIC']),
        'power_law_tail': fit_power_law_tail(values),
    }


fpt_data = read_rows(FPT_RESULT_FILE, 2)
if len(fpt_data) == 0:
    raise ValueError(f'No complete FPT records found in {FPT_RESULT_FILE}.')
fpt_times = fpt_data[:, 1]

summary_data = read_rows(RESULT_FILE, 3)
if len(summary_data):
    mfpt = summary_data[:, 1]
    completed = summary_data[:, 2]
else:
    mfpt, completed = reconstruct_mfpt(fpt_data)

analysis = {
    'configuration': {
        'N': N,
        'samples_per_shuffle_requested': SAMPLES_PER_SHUFFLE,
        'reshuffles_requested': RESHUFFLES,
        'complete_fpt_records': int(len(fpt_times)),
        'observed_landscapes': int(len(mfpt)),
        'mean_completed_trajectories_per_landscape': float(completed.mean()),
    },
    'individual_FPT': analyze('individual_FPT', fpt_times),
    'MFPT_across_landscapes': analyze('MFPT_across_landscapes', mfpt),
}

with (SCRIPT_DIR / 'Distribution_Fit_Report.json').open('w') as report:
    json.dump(analysis, report, indent=2)

with (SCRIPT_DIR / 'Distribution_Fit_Report.txt').open('w') as report:
    for section_name in ('individual_FPT', 'MFPT_across_landscapes'):
        section = analysis[section_name]
        report.write(f'{section_name}\n')
        report.write(f"n={section['sample_count']} mean={section['mean']:.8g} "
                     f"sd={section['standard_deviation']:.8g} median={section['median']:.8g}\n")
        report.write('Models ranked by AIC:\n')
        for fit in section['fits']:
            report.write(f"  {fit['distribution']}: AIC={fit['AIC']:.4f} "
                         f"BIC={fit['BIC']:.4f} KS={fit['KS']:.4f} "
                         f"KS_p={fit['KS_pvalue']:.4g} parameters={fit['parameters']}\n")
        report.write(f"Power-law tail: {section['power_law_tail']}\n\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
distribution_by_name = {
    'normal': stats.norm,
    'exponential': stats.expon,
    'gamma': stats.gamma,
    'weibull': stats.weibull_min,
    'lognormal': stats.lognorm,
}
for axis, values, title in zip(
    axes,
    (fpt_times, mfpt),
    ('Individual FPT', 'MFPT across shuffled landscapes'),
):
    sorted_values = np.sort(values[values > 0])
    survival = np.arange(len(sorted_values), 0, -1) / len(sorted_values)
    axis.loglog(sorted_values, survival, '.', markersize=3, alpha=0.45)
    section_name = 'individual_FPT' if title == 'Individual FPT' else 'MFPT_across_landscapes'
    fits = analysis[section_name]['fits']
    x_fit = np.logspace(np.log10(sorted_values.min()), np.log10(sorted_values.max()), 300)
    best_aic = fits[0]['AIC']
    for fit in fits[:3]:
        distribution = distribution_by_name[fit['distribution']]
        parameters = tuple(fit['parameters'].values())
        fitted_survival = distribution.sf(x_fit, *parameters)
        delta_aic = fit['AIC'] - best_aic
        label = f"{fit['distribution']} ($\\Delta$AIC={delta_aic:.1f})"
        axis.loglog(x_fit, fitted_survival, linewidth=2, label=label)
    axis.set_xlabel('Time / N')
    axis.set_ylabel('Empirical survival $P(T>t)$')
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / 'Distribution_Survival_Diagnostics.png', dpi=160, bbox_inches='tight')
plt.close(fig)

print(json.dumps(analysis, indent=2))
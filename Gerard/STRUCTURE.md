# Gerard project structure

This directory contains the main Julia research code, notebooks, input data, and
generated figures for the Gerard part of the project.

Use this file as the editable organization map. To request a future change,
describe it in [Pending changes](#pending-changes), then ask Copilot to read
this file and implement the unchecked items.

## Top-level directories

```text
Gerard/
|-- data/       Input tables and sequence data
|-- figures/    Generated figures and exported analysis tables
|-- notebooks/  Interactive analyses and grouped experiment runners
`-- src/        Julia source code
```

### `data/`

Input data used by simulations and analysis notebooks. This includes optimal
sequence CSV files and other reusable numerical inputs. Code should refer to
this directory through `DATA_DIR` from [`src/paths.jl`](src/paths.jl), rather
than through a working-directory-dependent relative path.

### `figures/`

Generated outputs. Figure folders are grouped by scientific topic and then by
analysis type:

```text
figures/
├── annealing/
│   ├── optimization/
│   ├── site/
│   └── target/
│       ├── dual/
│       ├── gillespie/
│       ├── no_stay/
│       ├── pdf_engineering/
│       ├── symmetric/
│       └── t1/
├── dynamics/
│   ├── frt/
│   ├── no_stay/
│   ├── ordering/
│   └── simulation/
├── landscape/
│   ├── exact/
│   ├── glassy/
│   ├── monte_carlo/
│   └── phase_transition/
├── moments/
│   ├── correlation/
│   ├── mean_field/
│   └── monte_carlo/
├── publication/
│   ├── abstract/
│   └── misc/
└── trajectories/
    └── ensemble/
```

New figure-producing code should use `figure_dir(...)` from
[`src/paths.jl`](src/paths.jl). This creates the directory automatically and
keeps output paths independent of where Julia or a notebook is launched.

### `src/`

Reusable and non-interactive Julia code, grouped by topic:

- [`src/annealing/`](src/annealing/): annealing optimization, site dynamics,
  target construction, and annealing input-data helpers.
- [`src/dynamics/`](src/dynamics/): first-return distributions, no-stay
  dynamics, ordering comparisons, and simulation/theory comparisons.
- [`src/landscape/`](src/landscape/): exhaustive landscape calculations,
  spin-glass phase-transition analysis, interactive construction, and
  publication-oriented landscape figures.
- [`src/moments/`](src/moments/): mean-field moment calculations and
  two-moment matching.
- [`src/trajectories/`](src/trajectories/): trajectory-ensemble analysis and
  packet/weight visualizations.
- [`src/testing/`](src/testing/): tests and verification scripts.
- [`src/paths.jl`](src/paths.jl): shared project-root, data, and figure-path
  helpers. Include this file when adding a script under `src/`.

### `notebooks/`

Interactive or exploratory work, grouped using the same topic names as `src/`:

- [`notebooks/annealing/`](notebooks/annealing/): site, target, and
  optimization analyses. [`annealing_optimization.ipynb`](notebooks/annealing/annealing_optimization.ipynb)
  runs the two related optimization scripts together.
- [`notebooks/landscape/`](notebooks/landscape/): exhaustive Cayley-landscape,
  minimum/phase-transition, and glassy-landscape analyses.
- [`notebooks/moments/`](notebooks/moments/): Monte Carlo scaling and moment
  relations.
- [`notebooks/trajectories/`](notebooks/trajectories/): packet detection and
  related trajectory analysis.
- [`notebooks/exploratory/`](notebooks/exploratory/): small or unfinished
  exploratory notebooks that do not yet belong to a larger workflow.

Notebook paths use `@__DIR__` and include
[`src/paths.jl`](src/paths.jl), so they can be opened and run without changing
the current working directory.

## How to request changes through this file

Add unchecked items below. Be specific about the current location, desired
location, and whether files or generated outputs must be preserved.

Recommended format:

```markdown
- [ ] Move `notebooks/example/analysis.ipynb` to `notebooks/topic/`.
  Update all figure paths and keep existing outputs.
- [ ] Merge `figures/topic_a/` and `figures/topic_b/` under
  `figures/topic/` with `topic_a/` and `topic_b/` subfolders.
- [ ] Combine the related scripts in `src/topic/` into a notebook runner.
```

When implementing a request, code and notebook paths must be updated together
with any moved figure folders. Existing files should be preserved unless a
change explicitly says that an artifact can be deleted.

## Pending changes

Add requested reorganizations here:

- [ ] No pending changes.

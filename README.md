Short description
-----------------
This repository contains code and notebooks for experimenting with second-order optimization
using a quadratic-approximation + annealer approach on small neural models (MLP). It includes
notebooks, example model state files and scripts for reproducing the experiments.

Quick start
-----------
- Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Open the main notebook and run cells to reproduce experiments:

```bash
jupyter notebook notebooks/main.ipynb
```

- Saved model states and configs are available in the `models/` directory.
- Statistics form the training are stored in `stats/` directory.

Notes
-----
- The code was developed for Python 3.10+. If you run into dependency issues, try updating
  package versions in `requirements.txt`.

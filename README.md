# Confounding-Valid Conformal Counterfactual Inference

This repository contains the code for running the experiments described in our work. The experiments are organized into two main directories: `Handover` and `Scheduling`.

## Repository Structure
- **`Handover/`**: Contains the scripts for the Handover experiments.
- **`Scheduling/`**: Contains the scripts for the Scheduling experiments.

## Conducting Experiments
To run an experiment, navigate to the relevant directory (`Handover` or `Scheduling`) and execute the python scripts located in the `model/` folder.

For example, to run a specific experiment sweep:
```bash
cd Handover
python model/run_sweep_eps.py
```

After running the experiments, the results will be saved in JSON format. You can then generate the plots by running the corresponding plotting scripts. Since there are multiple experiments and parameter sweeps, choose the script that matches your experiment:
- `python model/plot_sweep_eps.py` : Plots the results of the $\epsilon$ (epsilon) parameter sweep.
- `python model/plot_sweep_boxplot.py` (or `violin`) : Generates boxplots/violin plots for the sweep results.

Ensure to run the corresponding `run_sweep_*.py` script before attempting to generate its plots.

*Note: The generated plots and results are ignored by Git to keep the repository clean. Only the code is version-controlled.*

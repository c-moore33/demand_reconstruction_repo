# Household Water Demand Reconstruction

This repository contains the code and processed aggregate data supporting the manuscript *Estimating household water demand distributions reveals hidden dimensions of water utility performance* (Moore and Zaniolo).

The repository provides the code required to implement the  demand reconstruction framework and calculate the utility-performance metrics presented in the manuscript.

## Data availability and reproducibility

The full model incorporates proprietary household-level billing records obtained through utility partnerships. These records cannot be publicly distributed due to data-sharing restrictions. Consequently, only the model without household-level observations (version 6) can be run using the publicly available data. Model versions incorporating household observations cannot be fully reproduced from this repository, and postprocessing scripts specific to those restricted runs are not included.

## Repository structure

### 1. `preprocessing/`

Contains the cleaned aggregate data required to run the model and calculate utility-performance metrics. Primary data sources include the California Electronic Annual Report (eAR) and the 2023 American Community Survey (ACS).

### 2. `model/`

- `demand_reconstruction_model.py`: Defines the hierarchical Bayesian model structure.
- `run_model.py`: Executes the model and specifies the model configuration, including hyperparameters. Only version 6 can be run without proprietary household data. Model outputs are saved in NetCDF format.

### 3. `postprocessing/`

- `metric_calculations.py`: Calculates the utility-performance metrics presented in the manuscript. Requires model output in NetCDF format and the corresponding model training data.
- `figure_2.py`: Generates Figure 2, evaluating how well the inferred distributions reconstruct observed aggregate demand and revenue.

## Software requirements

The model is implemented in Python and relies on the following packages:

- PyMC
- ArviZ

## Running the model

1. Create and activate the Conda environment specified in `environment.yml`:

   ```bash
   conda env create -f environment.yml
   conda activate pymc_env
   ```

2. Verify the input data paths and model configuration in `run_model.py`.

3. Navigate to the `model` directory and run the aggregate-only model (version 6):

   ```bash
   cd model
   python run_model.py --version 6
   ```

4. The model output is saved as a NetCDF (`.nc`) file, which can be used alongside the corresponding training data to calculate utility-performance metrics using the scripts in `postprocessing/`.

## File paths

The following scripts may require modification of local file paths before execution:

- `utility_class.py`
- `run_model.py`
- `metric_calculations.py`
- `figure_2.py`

## Citation

If you use this framework, please cite:

Moore, C. & Zaniolo, M. *Estimating household water demand distributions reveals hidden dimensions of water utility performance*. Manuscript submitted.

[Publication DOI to be added upon publication.]


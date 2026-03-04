# martian-rampart-crater
Codes for analyzing Martian rampart crater ejecta

This repository contains the input files and Python analysis scripts associated with the paper:
**"Numerical modeling ejecta deposition from oblique impacts in icy layered terrains on Mars"** submitted to *Icarus*.

## 📌 Overview

Due to the large size of the raw numerical simulation outputs (generated via SALEc and SALEc-2D), the raw data files are not hosted in this repository. Instead, we provide the complete workflow to **reproduce the exact datasets and figures** presented in the manuscript.

This repository allows users to:
1. Re-run our impact simulations using the input files with SALEc or SALEc-2D.
2. Process the resulting ejecta data to recreate the figures shown in the paper (e.g., ejecta thickness distribution, shock pressure analysis).

## 🚀 How to Reproduce the Results

### Step 1: Run the Simulations
The simulations are conducted using the open-source hydrocodes **SALEc** and **SALEc-2D** (links provided in the main text of the manuscript). 
1. Install the hydrocode according to the authors' instructions.
2. Use the parameter files located in the `input_files/` directory to run the impact simulations for each target configuration.

### Step 2: Run the Analysis Scripts
Once the simulations are complete and the output data is generated on your local machine/server, you can use our Python scripts to process the ejecta data.

**Requirements:**
* Python 3.8+
* `numpy`
* `matplotlib`
* `pandas`
*  ......

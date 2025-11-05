# scWDAC
This repository contains the implementation of the scWDAC method for clustering single-cell multi-omics data, featuring both MATLAB and Python interfaces.

Overview
The scWDAC method is designed for clustering single-cell multi-omics data. It integrates multiple data views by employing a weighted distance penalty and adaptive consistent graph regularization to improve clustering performance.

Implementation Options
Option 1: Direct MATLAB Usage (Recommended for MATLAB Users)
Use the native MATLAB implementation for optimal performance.

Option 2: Python Interface with MATLAB Engine
Use the Python wrapper that calls the MATLAB implementation (requires MATLAB installation).
##################################################################################################################

Requirements
For MATLAB Usage:
MATLAB (R2020a or later recommended)
Required MATLAB toolboxes: Statistics and Machine Learning Toolbox
Quick Start:  MATLAB Example
Add the project directory to your MATLAB path:

% Open MATLAB and run the main script
scWDAC_main
This will run the scWDAC method on the example dataset (SNARE) and display the results.

Example 2: Using the function with your own data
Place your data file in the datasets folder:
Your data file should contain:
X: multi-view data (cell array)
true_label: ground truth labels (vector)
Modify the dataset name in scWDAC_main.m:

matlab
% Replace 'SNARE' with your dataset name
dataset = {'your_dataset_name'};

matlab
addpath('/path/to/scWDAC');
Load your data and run scWDAC:

#########################################################################################################################
For Python Usage:
1. Setup MATLAB Engine for Python
Check your MATLAB's supported Python version in the MATLAB directory under:
extern → engines → python

Create a compatible conda environment:

bash
conda create -n scwdac_env python=3.8
conda activate scwdac_env
Install MATLAB Engine for Python:

bash
cd "matlabroot/extern/engines/python"
python setup.py install
2. Run scWDAC
bash
python main.py



To make it easier for readers to run the program using Python, we have provided an interface that enables them to execute the scWDAC method through a Python program in PyCharm. The specific steps are as follows:

1. Check the Python version supported by your MATLAB.
Typically, you can find this information in the MATLAB installation directory under extern → engines → python. As shown in the image below, the example MATLAB version supports Python versions 3.8, 3.9, and 3.10.

Python (compatible with your MATLAB version; see setup instructions below)

MATLAB Engine for Python

Setup Instructions
1. Check MATLAB's Supported Python Version
Locate the supported Python version for your MATLAB installation. This can typically be found in your MATLAB directory under:
extern → engines → python

For example, if your MATLAB version supports Python 3.8, 3.9, and 3.10, ensure you use one of these versions.
<img width="334" height="203" alt="image" src="https://github.com/user-attachments/assets/426f8207-6a62-40f9-b63c-515a5d2123d6" />
2. Create Python Virtual Environment
Create a virtual environment using conda that matches the supported Python version:
<img width="266" height="177" alt="image" src="https://github.com/user-attachments/assets/ba2317bb-bf74-420b-bc33-fe57f305290c" />
<img width="416" height="79" alt="image" src="https://github.com/user-attachments/assets/edfb3605-f679-47cc-95a3-b757240b3d69" />

bash
conda create -n scwdac_env python=3.8
conda activate scwdac_env
Note: The scWDAC.m file must be located in the same project directory as the main.py file.


3. Install MATLAB Engine for Python
Navigate to the Python folder in your MATLAB installation directory (from Step 1) and install the MATLAB engine:

bash
cd "matlabroot/extern/engines/python"
python setup.py install
During installation, you will see various messages. If no red error messages appear and matlabengineforpython is displayed, the installation has completed successfully.
<img width="416" height="24" alt="image" src="https://github.com/user-attachments/assets/1c190ec0-a5fc-41a5-a831-f7fd9574a528" />
<img width="300" height="120" alt="image" src="https://github.com/user-attachments/assets/14c050d9-5637-4277-a217-741c30bf349c" />



4. Configure PyCharm (Optional)
If using PyCharm:

Activate the virtual environment created in Step 2

Ensure both scWDAC.m and main.py are in the same project directory

Usage
Activate your virtual environment:

bash
conda activate scwdac_env
Run the main Python script:

bash
python main.py
This will execute the scWDAC method using the provided example datasets.
<img width="318" height="143" alt="image" src="https://github.com/user-attachments/assets/aeaa8596-cb30-49fa-a8d6-df62cc1a3fdb" />

Datasets
Example datasets are provided in the datasets folder. Each dataset contains:
Multi-view data (X): Where each view X{v} has dimensions n × m (samples × features)
True labels (true_label): Ground truth labels for evaluation

File Structure
text
├── main.py                 # Python interface for running scWDAC
├── scWDAC.m               # Main MATLAB implementation
├── datasets/              # Example datasets
│   ├── SNARE.mat
│   └──  data1_sim.mat...
└── README.md             # This file



# scWDAC
This repository contains the implementation of the scWDAC method for clustering single-cell multi-omics data, featuring both MATLAB and Python interfaces.

Overview
The scWDAC method is designed for clustering single-cell multi-omics data. It integrates multiple data views by employing a weighted distance penalty and adaptive consistent graph regularization to improve clustering performance.

Implementation Options

Option 1: Direct MATLAB Usage (Recommended for MATLAB Users)

Use the native MATLAB implementation for optimal performance.

Option 2: Python Interface with MATLAB Engine

Use the Python wrapper that calls the MATLAB implementation (requires MATLAB installation).
########################################################

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

% Replace 'SNARE' with your dataset name 
dataset = {'your_dataset_name'};

Example Data Format
Your data file (e.g., your_data.mat) should contain:

% X: multi-view data (cell array)

X{1} = view1_data;  % n × m1 matrix

X{2} = view2_data;  % n × m2 matrix

true_label = [1,1,2,2,3,3,...];  % n × 1 vector

########################################################
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

<img width="334" height="203" alt="image" src="https://github.com/user-attachments/assets/2dd81be5-2a96-415d-b2f4-d5efb37d0a10" />


2. Create Python Virtual Environment
Create a virtual environment using conda that matches the supported Python version:
<img width="266" height="177" alt="image" src="https://github.com/user-attachments/assets/4367b788-a59d-4149-b145-fcf331e0192a" />

<img width="416" height="79" alt="image" src="https://github.com/user-attachments/assets/53a44524-b4c4-4919-a983-1f5e2eedd1ab" />

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


<img width="416" height="24" alt="image" src="https://github.com/user-attachments/assets/05b35431-a227-4143-804e-e656bf9bfe20" />

<img width="300" height="120" alt="image" src="https://github.com/user-attachments/assets/53514ae6-e039-4fce-b182-d9e2db939b6e" />


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

<img width="318" height="143" alt="image" src="https://github.com/user-attachments/assets/69c6b0fd-a2ad-43c9-818b-52b5fc97b51e" />

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



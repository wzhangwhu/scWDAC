# scWDAC

This repository contains the implementation of the scWDAC method for clustering single-cell multi-omics data, featuring both MATLAB and Python interfaces.

Overview
The scWDAC method is designed for clustering single-cell multi-omics data. It integrates multiple data views by employing a weighted distance penalty and adaptive consistent graph regularization to improve clustering performance.

Implementation Options

Option 1: Direct MATLAB Usage (Recommended for MATLAB Users)

Use the native MATLAB implementation for optimal performance.

Option 2: Python version of the code (Recommended for Python Users)
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
## 1. Environment Requirements
Before running the program, please ensure your environment meets the following requirements:
**Python version:** 3.8+ (recommended)
**Core dependency:** `numpy >= 1.26.4`.

## 2. Directory Structure
* `datasets/`: This folder contains the input data files (e.g., `.mat` files). Each dataset consists of multi-omic data (`X`), where the dimension of each omic `X{v}` is $n \times m$ (samples $\times$ features, e.g., number of cells $\times$ number of genes), along with the true labels (`true_label`).
* `functions/`: This directory stores the essential sub-functions and algorithm logic required for the scWDAC method.

## 3. Execution Steps
* Open the project in **PyCharm**.
* Locate and open the script **`scWDAC_main.py`**.
* **Configure the dataset:** On **Line 32** of `scWDAC_main.py`, you can specify the dataset you wish to process by modifying the filename in the following line:
   ```python
   data_path = os.path.join("datasets", "data1_sim.mat")
* Run the script **`scWDAC_main.py`**.

* 
<p align="left">
  <img src="scWDAC_Python/figure/simu1_result.png" width="600px" alt="Simulation Result">
</p>



## 4. Output and Evaluations
Once the execution is complete, the program will output a result vector. This vector contains the following performance metrics in order:
| Index | Metric | Description |
| :--- | :--- | :--- |
| 1 | **ACC** | Accuracy |
| 2 | **NMI** | Normalized Mutual Information |
| 3 | **F-score** | F1 Score |
| 4 | **Precision** | Precision |
| 5 | **ARI** | Adjusted Rand Index |
| 6 | **Purity** | Purity |
| 7 | **Recall** | Recall |
| 8 | **Entropy** | Entropy |
| 9 | **Time** | Time (seconds) |



# scWDAC

**Source code for "Clustering Single-Cell Multi-Omics Data via Weighted Distance Penalty and Adaptive Consistent Graph Regularization"**

This repository provides the MATLAB implementation of the **scWDAC** method for clustering single-cell multi-omics data.

---

## Overview
The scWDAC method is designed for clustering single-cell multi-omics data. It integrates multiple data views by employing a weighted distance penalty and adaptive consistent graph regularization to improve clustering performance.

## **1. Environment Requirements**
Before running the program, please ensure your environment meets the following requirements:
* **MATLAB Version:** R2020a or later recommended.

## **2. Directory Structure**
* `datasets/`: This folder contains the input data files (e.g., `.mat` files). Each dataset consists of multi-omic data (`X`), where the dimension of each omic `X{v}` is $n \times m$ (samples $\times$ features, e.g., number of cells $\times$ number of genes), along with the true labels (`true_label`).
* `functions/`: This directory stores the essential sub-functions and algorithm logic required for the scWDAC method.

## **3. Execution Steps**
* Add the project directory to your MATLAB path.
* Open MATLAB and run the main script:
   ```matlab
   scWDAC_main

## **3. Execution Steps**
* Open the project in **MATLAB**.
* Locate and open the script **`scWDAC_main.m`**.
* **Configure the dataset:** On **Line 13** of `scWDAC_main.m`, you can specify the dataset you wish to process by modifying the filename in the following line:
   ```matlab
   dataset = {'data1_sim'}
* Run the script **`scWDAC_main.m`**.
<p align="left">
  <img src="figure/simu1_result.png" width="600px" alt="Simulation Result">
</p>

## **4. Output and Evaluations**
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

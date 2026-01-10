# scWDAC

**Source code for "Clustering Single-Cell Multi-Omics Data via Weighted Distance Penalty and Adaptive Consistent Graph Regularization"**

To make it easier for readers to run the program using Python, we have provided an interface that enables them to execute the **scWDAC** method through a Python program in PyCharm. 

---

## Overview
The scWDAC method is designed for clustering single-cell multi-omics data. It integrates multiple data views by employing a weighted distance penalty and adaptive consistent graph regularization to improve clustering performance.

## 1. Environment Requirements
Before running the program, please ensure your environment meets the following requirements:
* **Python version:** 3.8+ (recommended)
* **Core dependency:** `numpy >= 1.26.4`.

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
<p align="left">
  <img src="figure/simu1_result.png" width="600px" alt="Simulation Result">
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


# student-performance-ml-algorithms
Implementation and comparison of 19 Machine Learning algorithms using the UCI Student Performance Dataset.
# Machine Learning Assignment – Student Performance

## Project Overview

This project implements **19 Machine Learning algorithms** using the **UCI Student Performance Dataset (`student-mat.csv`)**.

The complete assignment is contained in a single Python program:

```text
algorithms.py
```

The program loads the dataset, preprocesses the data, runs all 19 algorithms, evaluates them, and prints a final summary.

## Algorithms Implemented

1. K-Means
2. Modified K-Means
3. Hierarchical Clustering
4. Fuzzy C-Means
5. DBSCAN
6. HDBSCAN
7. Self-Training
8. Random Forest Classification
9. Random Forest Regression
10. XGBoost
11. AdaBoost
12. CatBoost
13. MLP
14. RNN
15. SOM
16. HMM
17. SVM
18. LLM
19. GRNN

## Dataset

**Student Performance Dataset – `student-mat.csv`**

The final grade (`G3`) is used as the target.

For classification:

```text
G3 >= 10 → Pass
G3 < 10  → Fail
```

For regression, the numerical `G3` value is predicted.

## Project Files

```text
ML_assignment/
│
├── algorithms.py
├── student-mat.csv
├── requirements.txt
└── README.md
```

## Requirements

Recommended Python version:

```text
Python 3.13 (64-bit)
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

## How to Run

Open PowerShell in the project folder and run:

```powershell
python algorithms.py
```

The program will automatically run all 19 algorithms and display the results.

## Example Final Output

```text
============================================================
FINAL MACHINE LEARNING SUMMARY
============================================================

Algorithm                    Result             Evaluation
--------------------------------------------------------------------
K-Means                      Clustering         Silhouette: 0.0777
Modified K-Means             Clustering         Silhouette: 0.0726
Hierarchical                 Clustering         Silhouette: 0.0552
Fuzzy C-Means                Clustering         Silhouette: 0.0871
DBSCAN                       Clustering         Silhouette: 0.4148
HDBSCAN                      Clustering         Silhouette: 0.1630
Self-Training                Classification     Accuracy: 63.87%
Random Forest Classifier     Classification     Accuracy: 70.59%
Random Forest Regression     Regression         RMSE: 3.9157, R²: 0.3025
XGBoost                      Classification     Accuracy: 64.71%
AdaBoost                     Classification     Accuracy: 69.75%
CatBoost                     Classification     Accuracy: 73.11%
MLP                          Classification     Accuracy: 68.07%
RNN                          Classification     Accuracy: 68.07%
SOM                          Clustering         Quantization Error: 2.5028
HMM                          Sequence Model     Accuracy: 71.90%
SVM                          Classification     Accuracy: 68.07%
LLM                          Language Task      Summary Generated
GRNN                         Regression         RMSE: 4.4673, R²: 0.0921
--------------------------------------------------------------------

Total Algorithms: 19
Successful: 19
Failed: 0
```

## Evaluation Metrics

Different algorithms use different evaluation metrics:

* **Accuracy** – classification performance
* **Silhouette Score** – cluster quality
* **ARI** – agreement between clusters and reference classes
* **RMSE** – regression prediction error
* **R²** – regression explanatory power
* **Quantization Error** – SOM representation quality

## Notes

* All 19 algorithms are executed from **one Python file**.
* The program does not use artificially generated student data.
* Results may vary slightly between runs because some algorithms use random initialization.
* The LLM component uses a pretrained language model and does not train an LLM on the student dataset.

## Expected Result

A successful execution should finish with:

```text
Total Algorithms: 19
Successful: 19
Failed: 0
```

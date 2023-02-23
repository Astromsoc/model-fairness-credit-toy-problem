# A Toy Implementation of Model Fairness Measurement, Discussion & Improvement

### Toy Dataset: German Credit

_available here: https://www.kaggle.com/code/kabure/predicting-credit-risk-model-pipeline/data_

--


### Navigation

This repo features the following structure:

```
.
├── LICENSE
├── README.md
├── codes
│   ├── 1_global_surrogate.py
│   ├── 2_fairness_measurement.ipynb
│   ├── 3_fairness_improvement.ipynb
│   ├── __init__.py
│   ├── const.py
│   └── utils.py
├── data
│   ├── labeled_train.csv
│   ├── labeled_train_augmented.csv
│   └── unlabeled_test.csv
├── requirements.txt
├── results
│   ├── LDA-roc-auc.png
│   ├── LR-roc-auc.png
│   ├── RF-roc-auc.png
│   ├── anti-classification-age.png
│   ├── anti-classification-sex.png
│   ├── group-fairness-age.png
│   ├── group-fairness-sex.png
│   ├── model-comparison.png
│   ├── new-distribution-gender.png
│   ├── separation-1-before.png
│   ├── separation-2-after.png
│   ├── separation-age.png
│   └── separation-sex.png
└── streamline.sh
```

where the major folders are introduced below:

(1) `codes` 

Scripts and notebooks for different purposes.

`const.py`: all needed constant variables

`utils.py`: self-defined functions

`1_global_surrogate.py`: to train and compare several models chosen

`2_fairness_measurement.ipynb`: to measure fairness from 3 perspectives: anti-classification, group fairness and separation.

`3_fairness_improvement.ipynb`: to improve fairness through exclusion of protected attributes, adaptive thresholds and data augmentation, with comparison on model performance and fairness measures before and after each method tried.

(2) `data`

Original (and augmented, after execution of `3_fairness_improvement.ipynb`) data files.

(3) `results`

All graphs generated in the process.


--


### Guidance on Execution

__Step 1__ package installation

Run:

`pip3 install -r requirements.txt`

in the root directory.


__Step 2__ model training and comparison

Run:

`bash streamline.sh`

in the root directory, which runs `./codes/1_global_surrogate.py`.

__Step 3__ fairness measurement

Execute `./codes/2_fairness_measurement.ipynb`.

__Step 4__ fairness improvement

Execute `./codes/3_fairness_improvement.ipynb`.


# Credit Card Fraud Detection
### A Machine Learning Approach to Financial Anomaly Detection

---

## Overview

This project presents a supervised machine learning 
pipeline designed to detect fraudulent credit card 
transactions within a dataset of 1,296,675 records. 
The work was undertaken as part of the CSI GRIET 
Technical Recruitment — Round 2, with the objective 
of demonstrating not merely technical implementation, 
but a thorough understanding of data, methodology, 
and analytical reasoning.

The central challenge of this domain is not complexity 
of algorithms — it is the severe class imbalance 
inherent to fraud data. With fraudulent transactions 
comprising only 0.58% of all records, conventional 
approaches fail systematically. This project addresses 
that challenge at every stage of the pipeline.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Analysis](#dataset-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Selection & Justification](#model-selection--justification)
- [Class Imbalance Strategy](#class-imbalance-strategy)
- [Evaluation Methodology](#evaluation-methodology)
- [Results & Analysis](#results--analysis)
- [Model Limitations](#model-limitations)
- [Future Directions](#future-directions)
- [How to Run](#how-to-run)
- [Dataset Note](#dataset-note)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Fraud detection presents a deceptively difficult 
classification problem. A naive model that classifies 
every transaction as legitimate achieves 99.42% accuracy 
on this dataset — yet fails to identify a single 
fraudulent transaction. This exposes the fundamental 
inadequacy of accuracy as an evaluation metric in 
imbalanced classification tasks.

The objective is to construct a system that 
distinguishes fraudulent transactions from legitimate 
ones with high precision and recall, evaluated through 
metrics appropriate to the domain: Precision, Recall, 
F1-Score, and ROC-AUC.

The consequences of each type of error are asymmetric:
- **False Negatives** — Fraudulent transactions 
  classified as legitimate, allowing financial harm 
  to the cardholder.
- **False Positives** — Legitimate transactions 
  flagged as fraud, eroding customer trust and 
  disrupting normal commerce.

A production-grade system must manage this tradeoff 
deliberately, not accidentally.

---

## Dataset Analysis

The dataset comprises 1,296,675 transaction records 
across 26 features, provided directly by CSI GRIET 
as a modified version of a publicly available 
Kaggle dataset.

**Initial exploratory analysis revealed several 
noteworthy characteristics:**

**Severe Class Imbalance**  
7,506 fraudulent transactions exist among 1,289,169 
legitimate ones — a ratio of approximately 1:172. 
Any model trained without addressing this imbalance 
will exhibit strong bias toward the majority class.

**Deliberate Noise Columns**  
Two columns — `random_noise_1` and `random_noise_2` 
— were identified as containing entirely random, 
meaningless values with no correlation to the target 
variable. Their presence appears intentional, designed 
to assess whether candidates exercise critical judgment 
during feature selection rather than blindly including 
all available columns.

**Missing Value Distribution**  
- `merch_zipcode`: 195,973 missing values (15.1% 
  of dataset) — column removed entirely
- `category`, `amt`, `city_pop`: 64,834 missing 
  values each — addressed through imputation

---

## Data Preprocessing

All preprocessing decisions were made with explicit 
justification rather than convention:

**Columns Removed**

| Column | Reason |
|--------|--------|
| `cc_num`, `first`, `last`, `street`, `trans_num` | Unique identifiers — inclusion causes data leakage, training the model to memorise individuals rather than learn fraud patterns |
| `random_noise_1`, `random_noise_2` | Confirmed irrelevant through statistical analysis |
| `merch_zipcode` | Excessive missing values render the column statistically unreliable |
| `Unnamed: 0` | Index artifact with no predictive value |

**Missing Value Treatment**
- `category` — Imputed with mode (categorical variable)
- `amt` — Imputed with median (robust to outliers)
- `city_pop` — Imputed with median (skewed distribution)

**Categorical Encoding**  
Label Encoding applied to: `merchant`, `category`, 
`gender`, `city`, `state`, `job`

**Feature Scaling**  
StandardScaler applied to all numerical features. 
Critically, the scaler was fitted exclusively on 
training data and applied to test data — preventing 
any information leakage from the test set into 
preprocessing.

---

## Feature Engineering

Feature engineering represents the most analytically 
significant contribution of this project. Rather than 
relying solely on existing columns, three domain-informed 
features were constructed to capture fraud signals 
not present in the raw data.

### Transaction Distance
**Method:** Haversine formula applied to cardholder 
GPS coordinates (`lat`, `long`) and merchant GPS 
coordinates (`merch_lat`, `merch_long`)  
**Output:** Great-circle distance in kilometres  
**Rationale:** Geographic impossibility is one of 
the most reliable fraud indicators. A transaction 
occurring thousands of kilometres from the 
cardholder's registered location within a short 
timeframe represents a physical impossibility 
under normal circumstances.

### Transaction Hour
**Method:** Hour extracted from `trans_date_trans_time`  
**Output:** Integer value 0–23  
**Rationale:** Temporal analysis of the dataset 
reveals a consistent pattern — fraudulent 
transactions are significantly overrepresented 
between 00:00 and 04:00 hours, when cardholders 
are statistically least likely to be actively 
monitoring their accounts.

### Customer Age
**Method:** Age calculated from `dob` relative 
to transaction date  
**Output:** Age in years  
**Rationale:** Demographic context meaningfully 
influences spending behaviour baselines. 
Age provides the model with contextual grounding 
for what constitutes an anomalous transaction 
for a given individual.

---

## Model Selection & Justification

Two models were selected deliberately to represent 
contrasting points on the interpretability-performance 
spectrum.

### Logistic Regression — Interpretable Baseline
Logistic Regression was implemented as a statistical 
baseline. Its value lies not in performance ceiling 
but in transparency — its coefficients are directly 
interpretable, and its results establish a reference 
point against which the complexity of XGBoost 
can be meaningfully evaluated.

A complex model that cannot substantially outperform 
a linear baseline raises legitimate questions about 
whether that complexity is warranted.

### XGBoost — Primary Detection Model
XGBoost operates through gradient boosting — 
constructing an ensemble of decision trees 
sequentially, where each successive tree is 
trained specifically to correct the residual 
errors of its predecessors. This architecture 
enables the model to capture non-linear 
relationships and complex feature interactions 
that linear models cannot represent.

XGBoost is the industry standard for tabular 
fraud detection, deployed by major financial 
institutions and payment processors globally. 
Its selection here reflects both its proven 
performance characteristics and its suitability 
for imbalanced classification tasks when 
appropriately configured.

---

## Class Imbalance Strategy

The 1:172 class ratio required explicit intervention. 
Two complementary strategies were employed:

**SMOTE (Synthetic Minority Oversampling Technique)**  
SMOTE generates synthetic fraud examples by 
interpolating between existing fraud instances 
in feature space, rather than simply duplicating 
them. This provides the model with sufficient 
exposure to fraud patterns during training without 
introducing the overfitting risks of naive duplication.

Application was restricted strictly to training data. 
Applying SMOTE to test data would corrupt evaluation 
metrics by introducing synthetic examples into 
what should represent real-world conditions.

**scale_pos_weight (XGBoost)**  
An additional class weight parameter was passed 
to XGBoost, calculated as the ratio of negative 
to positive training examples. This penalises 
misclassification of the minority class more 
heavily during training, further counteracting 
the imbalance.

---

## Evaluation Methodology

Four metrics were used to evaluate model performance:

| Metric | What It Measures |
|--------|-----------------|
| **Precision** | Of all transactions flagged as fraud, what proportion are genuinely fraudulent? |
| **Recall** | Of all genuinely fraudulent transactions, what proportion were correctly identified? |
| **F1-Score** | Harmonic mean of Precision and Recall — the primary metric for imbalanced classification |
| **ROC-AUC** | The model's overall ability to discriminate between classes across all classification thresholds |

Accuracy was deliberately excluded from primary 
evaluation for reasons established in the 
Problem Statement.

---

## Results & Analysis

| Metric    | Logistic Regression | XGBoost  |
|-----------|---------------------|----------|
| Precision | 0.07                | **0.87** |
| Recall    | 0.73                |   0.69   |
| F1-Score  | 0.13                | **0.77** |
| ROC-AUC   | 0.8313              |**0.9815**|

**XGBoost demonstrates decisive superiority 
across all primary metrics.**

The precision differential is particularly 
significant. Logistic Regression's precision 
of 0.07 indicates that 93 of every 100 fraud 
alerts generated are false positives — 
operationally unacceptable in any production 
context. XGBoost's precision of 0.87 
represents a 12-fold improvement.

Logistic Regression achieved marginally 
superior recall (0.73 vs 0.69), reflecting 
the classical precision-recall tradeoff — 
its tendency to flag aggressively increases 
fraud capture at the expense of an 
unacceptable false positive rate.

XGBoost's ROC-AUC of 0.9815 indicates 
near-excellent discriminative capability 
across all classification thresholds, 
confirming its suitability as the 
primary detection model.

---

## Model Limitations

Intellectual honesty demands an examination 
of where the system falls short.

XGBoost's recall of 0.69 indicates that 
31% of fraudulent transactions were not 
detected. Analysis of false negatives 
reveals a consistent pattern: the 
majority involve low-value transactions 
that closely resemble legitimate 
everyday purchases.

This reflects a known fraud strategy — 
perpetrators frequently initiate stolen 
card usage with small test transactions 
to verify card validity before escalating 
to high-value purchases. Without 
time-window velocity features, the model 
evaluates each transaction in isolation 
and cannot detect this sequential pattern.

This limitation is architectural rather 
than parametric — it cannot be resolved 
through hyperparameter tuning alone, 
but requires additional feature engineering.

---

## Future Directions

**Velocity Feature Engineering**  
Constructing rolling time-window features — 
transaction frequency per card over intervals 
of 5, 15, and 60 minutes — would directly 
address the sequential fraud pattern 
identified in the limitations analysis.

**Hyperparameter Optimisation**  
The current XGBoost implementation uses 
conservative default parameters. 
Systematic optimisation via RandomizedSearchCV 
across `max_depth`, `learning_rate`, 
`n_estimators`, and `subsample` parameters 
would likely yield measurable performance gains.

**Ensemble Architecture**  
Combining XGBoost with LightGBM through 
a voting or stacking ensemble could further 
reduce the false positive rate by requiring 
consensus between two independent 
high-performance models before flagging 
a transaction.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/ronaldjose-ship-it/credit-card-fraud-detection.git

# Navigate to project directory
cd credit-card-fraud-detection

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter

# Place final_dataset.csv in the project directory

# Launch the notebook
jupyter notebook fraud_detection.ipynb
```

---

## Dataset Note

The dataset was provided directly by CSI GRIET 
for recruitment purposes — a modified version 
of the following publicly available resource, 
with deliberate alterations including noise 
column insertion and row count reduction:

https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset

---

## Tech Stack

| Library | Application |
|---------|------------|
| Python 3.13 | Core language |
| Pandas & NumPy | Data manipulation and numerical computing |
| Matplotlib & Seaborn | Statistical visualisation |
| Scikit-learn | Preprocessing, Logistic Regression, evaluation metrics |
| XGBoost | Primary gradient boosting classifier |
| Imbalanced-learn | SMOTE implementation |
| Jupyter Notebook | Development and presentation environment |

---

## Conclusion

The most important decisions in this project were not 
made in code — they were made before a single line 
was written.

**Supervised vs Unsupervised**  
When the dataset revealed explicit fraud labels, the 
choice of Supervised Learning was not a preference 
— it was a logical necessity. An unsupervised approach 
would have searched for statistical outliers without 
understanding what fraud actually looks like. Supervised 
Learning granted the models something far more valuable 
than pattern recognition — it granted them definition. 
They did not search for the unusual. They searched for 
the fraudulent. That distinction determined everything 
that followed.

**Logistic Regression vs XGBoost**  
Both models received identical data, identical 
preprocessing, and identical opportunities to learn. 
What separated them was not effort — it was capability. 
Logistic Regression drew a straight line through a 
problem that does not have straight lines. Fraud hides 
in the intersections — a small transaction, at 3 AM, 
from a merchant 2,000 kilometres away. No single feature 
condemns it. The combination does. Logistic Regression 
cannot see combinations. XGBoost was built for them.

The result speaks for itself — 0.07 precision against 
0.87. Ninety-three false accusations for every seven 
criminals caught, against eight criminals caught for 
every one innocent person inconvenienced.

This project reinforced a principle that transcends 
machine learning: understanding the problem deeply 
always matters more than solving it quickly.

---
*Submitted for CSI GRIET Technical Recruitment — Round 2*
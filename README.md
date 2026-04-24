# 🛡️ Credit Card Fraud Detection — Machine Learning

Every 2 seconds a fraud attempt happens somewhere in the world, leaving a trail of financial devastation. This project isn't just about writing code; it builds a predictive system designed to catch these invisible thieves before they strike.

## 📑 Table of Contents
- [Problem Statement](#-problem-statement)
- [The Dataset](#-the-dataset)
- [Feature Engineering (The Secret Sauce)](#️-feature-engineering-the-secret-sauce)
- [Approach & Reasoning](#-approach--reasoning)
- [Models Used](#-models-used)
- [Results Table](#-results-table)
- [Key Insights](#-key-insights)
- [Where the Model Failed](#️-where-the-model-failed)
- [Future Improvements](#-future-improvements)
- [How to Run](#-how-to-run)
- [Dataset Note](#-dataset-note)
- [Tech Stack](#-tech-stack)

## 🚨 Problem Statement
Fraud detection is like finding a needle in a digital haystack. Millions of legitimate transactions happen daily, but only a tiny fraction are fraudulent. This project tackles the immense challenge of **class imbalance**—where normal transactions effectively drown out the fraudulent ones. 

If a model simply guesses "Not Fraud" every single time, it achieves an *accuracy* of 99.42%. It looks fantastic on paper, but it actively allows every single criminal straight through the gates. This is why accuracy is a useless metric here, and the focus is uniquely locked onto Precision, Recall, and F1-Score.

## 📊 The Dataset
This project utilises a transaction dataset holding **1.3M records**, out of which only **0.58% are actual fraud**.
- **Noise Detection**: Initial EDA discovered and removed completely irrelevant columns like `random_noise_1` and `random_noise_2`.
- **Missing Value Handling**: Dropped `merch_zipcode` entirely as it was intensely corrupted (195,973 missing values), while imputing `category` with its mode, and continuous variables like `amt` and `city_pop` with robust median distributions.

## 🛠️ Feature Engineering (The Secret Sauce)
Building models is inherently easy; feeding them the exact right information is the hard part. I engineered several unique features to give the algorithms an edge:

- **`transaction_distance`**: Utilizing the Haversine formula, this calculates the geographical GPS distance between the cardholder and the merchant. Intuition: If a card is used in New York and 5 minutes later in London, it's a physical impossibility.
- **`trans_hour`**: Isolated and extracted the exact hour of the transaction. Intuition: Normal shopping happens during standard business hours; fraudsters wait until you are asleep.
- **`customer_age`**: Calculated directly from embedded birth dates. Intuition: Different demographics exhibit vastly different baseline spending behaviors, and elder victims are targeted in fundamentally different ways than younger users.

## 🧠 Approach & Reasoning
- **Why Supervised Learning?** Since every baseline transaction in the historical data is explicitly labelled, Supervised Learning is the mathematically optimal framework to deploy.
- **Why these exact models?** I utilized Logistic Regression to act as a highly-interpretable, fast statistical baseline. That baseline was then challenged by XGBoost, an industry-standard algorithm designed to carve through complex, non-linear logic.
- **What SMOTE actually does**: Because fraud is exceptionally rare, the training parameters become biased. SMOTE (Synthetic Minority Over-sampling Technique) maps the vectors of known fraud and geometrically creates highly realistic, synthetic fraud data. This allows the model to learn the true shape of a thief without being overwhelmed by millions of everyday transactions.

## 🤖 Models Used
- **Logistic Regression**: A simple, interpretable algorithm that draws a clean statistical line between legitimate logic and fraud.
- **XGBoost**: A powerful, industry-standard ensemble algorithm. It functions by building successive decision trees where each new tree directly attempts to correct the errors of the previous one.

## 📈 Results Table

| Metric    | Logistic Regression | XGBoost |
|-----------|-------------------- | --------|
| Precision |       0.07          |  0.87   |
| Recall    |       0.73          |  0.69   |
| F1-Score  |       0.13          |  0.77   |
| ROC-AUC   |      0.8313         |  0.9815 |

## 💡 Key Insights
- **Night Owls**: Fraud doesn't clock in at 9 AM; it peaks heavily in the dead hours between 12 AM and 4 AM.
- **Geographical Flags**: High calculated transaction distance is one of the strongest, most undeniable signals of card theft.
- **Testing the Waters**: Low-amount fraud is by far the hardest anomaly to confidently catch, as criminals test stolen cards using completely innocent-looking, small-value purchases.

## ⚠️ Where the Model Failed
Honesty is crucial in any robust data scientific analysis. The model actively struggles to flag isolated low-amount transactions. These micro-purchases perfectly mimic daily shopping (like buying a coffee). Because the model is not tracking the immediate velocity (the rapid repeating rate) of these micro-transactions, small individual hits occasionally slip safely beneath the threshold. 

## 🚀 Future Improvements
The system sets a highly powerful foundation, but there is always room to escalate the defense:
- **Hyperparameter tuning**: Systematically maximizing the XGBoost potential utilizing Grid Search architecture.
- **Ensemble methods**: Layering XGBoost with other specialized classifiers like LightGBM to further harden the false-positive barrier.
- **Velocity features**: Engineering rolling time-window transaction counts (e.g. 'transactions per hour per card') to immediately shut down rapid-fire testing sequences.

## 💻 How to Run
Follow these simple steps:
1. Clone the repo
2. Open your terminal in the directory
3. Install requirements using `pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn`
4. Launch Jupyter Notebook and open `fraud_detection.ipynb`
*(Note: Ensure `final_dataset.csv` is located in the same directory.)*


## 📝 Dataset Note
Dataset provided directly by CSI club (modified 
version of original Kaggle dataset).

Original dataset available at:
https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset

## 🛠 Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, SMOTE, Matplotlib, Seaborn

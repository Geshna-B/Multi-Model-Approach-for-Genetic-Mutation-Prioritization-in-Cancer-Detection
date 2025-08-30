# Multi-Model Approach for Genetic Mutation Prioritization in Cancer Detection  

## Overview  
This project explores an AI-based system for classifying genetic mutations using clinical text data. By applying multiple preprocessing techniques and machine learning models, the system prioritizes mutations into **nine cancer-related classes** with high accuracy.  



## Objectives  
- Preprocess clinical text data (cleaning, tokenization, vectorization).  
- Implement and compare ML models:  
  - Random Forest  
  - Logistic Regression  
  - XGBoost  
  - LightGBM  
- Evaluate performance using **Accuracy, Precision, Recall, F1-score, and ROC-AUC**.  
- Identify the best model-preprocessing combination for mutation classification.  



## Methodology  
- **Data Source**: Kaggle genetic mutation dataset (`training_variants.csv`, `training_text.csv`).  
- **Preprocessing**: CountVectorizer, TF-IDF, Word2Vec, Doc2Vec.  
- **Models**: Random Forest (RF), Logistic Regression (LR), XGBoost, LightGBM.  
- **Evaluation**: Cross-validation, confusion matrix, ROC-AUC.  
- **Best Result**: **LightGBM + Doc2Vec → 93% accuracy**.  



## Key Findings  
- **LightGBM** consistently outperformed other models.  
- **Doc2Vec** embeddings provided the most meaningful text representation.  
- Traditional models like Logistic Regression and Random Forest were less effective.  



## Installation & Usage  

### 1. Clone Repository  

git clone https://github.com/your-username/genetic-mutation-classification.git
cd genetic-mutation-classification


### 2. Install Dependencies


pip install -r requirements.txt


**Main dependencies**:

* `numpy`, `pandas`, `scikit-learn`
* `xgboost`, `lightgbm`
* `gensim` (for Word2Vec/Doc2Vec)
* `matplotlib`, `seaborn` (for visualization)

### 3. Dataset

Download the dataset from Kaggle:
👉 [Kaggle – Redefining Cancer Treatment](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data)

### 4. Run Preprocessing


python preprocess.py


This will clean, tokenize, and vectorize clinical text using the selected method (`CountVectorizer`, `TF-IDF`, `Word2Vec`, or `Doc2Vec`).

### 5. Train & Evaluate Models


python train.py --model lightgbm --vectorizer doc2vec


**Options**:

* `--model` → `random_forest`, `logistic_regression`, `xgboost`, `lightgbm`
* `--vectorizer` → `count`, `tfidf`, `word2vec`, `doc2vec`

### 6. View Results

Outputs include:

* Accuracy, Precision, Recall, F1-Score
* Confusion Matrix
* ROC-AUC Curve



## 🚀 Future Work

* Hyperparameter tuning with Bayesian optimization.
* Integration of advanced embeddings (BERT, BioBERT, ClinicalBERT).
* Multimodal data fusion (genetic sequences + clinical evidence).
* Explainability via SHAP/LIME for clinical adoption.



## 👩‍💻 Team
* Geshna B
*  Katikala Dedeepya
* Malavika S Prasad
* Vada Gouri Hansika Reddy



**Supervised by**: Dr. S.S. Kalaivani & Dr. Manoj Bhatt K

**Affiliation**: Amrita Vishwa Vidyapeetham, Department of AI

# 📊 Stock Sentiment Analysis Using News Headlines

Predict whether a stock's price will **rise (1)** or **fall (0)** on a given day using the **top 25 news headlines** — powered by NLP and Machine Learning.

## 🎯 Objective

Build a binary classification system that analyzes daily news headlines to predict stock market direction. The project compares multiple ML pipelines to find the best-performing model.

---

## 📁 Dataset

| Property | Details |
|----------|---------|
| **Source** | Historical news headlines dataset (`Data.csv`) |
| **Records** | 3,992 trading days (2000-01-03 to 2016-01-27) |
| **Features** | 25 news headlines per day (`Top1` to `Top25`) |
| **Target** | `Label` — `1` (stock price went up), `0` (stock price went down) |

### Sample Data

| Date | Label | Top1 | Top2 | ... | Top25 |
|------|-------|------|------|-----|-------|
| 2000-01-03 | 0 | A 'hindrance to operations'... | Scorecard | ... | Recovering a title |
| 2000-01-06 | 1 | Pilgrim knows how to progress | Thatcher facing ban | ... | Lessons of law's hard heart |

---

## 🔄 Pipeline

```
Raw Headlines → Text Preprocessing → Vectorization → Model Training → Prediction
```

### 1. Text Preprocessing
- Removed all **punctuation and special characters** using regex
- Converted all text to **lowercase**
- **Concatenated** 25 headlines per day into a single document

### 2. Train/Test Split
- **Training set:** All data before 2015 (~3,723 days)
- **Test set:** All data from 2015 onwards (**269 days**)

### 3. Feature Extraction
- **CountVectorizer** — Bi-gram bag of words (`ngram_range=(2,2)`)
- **TF-IDF Vectorizer** — Bi-gram TF-IDF weighting (`ngram_range=(2,2)`)

### 4. Models Trained
- Random Forest Classifier (200 trees, entropy criterion)
- Multinomial Naive Bayes

---

## 📈 Results

| # | Pipeline | Accuracy | F1-Score | Precision (0/1) | Recall (0/1) |
|---|----------|----------|----------|-----------------|--------------|
| 1 | CountVectorizer + Random Forest | 95.91% | 0.96 | 0.98 / 0.94 | 0.94 / 0.98 |
| 2 | TF-IDF + Random Forest | 96.28% | 0.96 | 0.97 / 0.95 | 0.96 / 0.97 |
| 3 | **TF-IDF + Naive Bayes** | **96.65%** | **0.97** | **1.00 / 0.94** | **0.94 / 1.00** |

### 🏆 Best Model: TF-IDF + Multinomial Naive Bayes

```
Confusion Matrix:
              Predicted 0    Predicted 1
Actual 0:        130              9
Actual 1:          0            130

Accuracy:  96.65%
F1-Score:  0.97
```

**Key highlights:**
- ✅ **100% recall** on positive class — zero missed upward movements
- ✅ Only **9 misclassifications** out of 269 test samples
- ✅ Simplest model outperformed Random Forest

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3 |
| **Libraries** | Pandas, Scikit-learn |
| **NLP Techniques** | Bag of Words, TF-IDF, Bi-gram Tokenization |
| **ML Algorithms** | Random Forest, Multinomial Naive Bayes |
| **Environment** | Google Colab / Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn
```

### Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/rhythmsharma2004/Stock-Sentiment-Analysis-using-News-Headlines.git
   cd Stock-Sentiment-Analysis-using-News-Headlines
   ```
2. Place `Data.csv` in the project directory
3. Open and run the notebook:
   ```bash
   jupyter notebook "_Stock_Sentiment_Analysis_using_News_Headlines.ipynb"
   ```

---

## 📂 Project Structure

```
Stock-Sentiment-Analysis-using-News-Headlines/
│
├── Stock_Sentiment_Analysis_using_News_Headlines.ipynb   # Main notebook
├── Data.csv                                               # Dataset
└── README.md                                              # Project documentation
```

---

## 💡 Key Takeaways

1. **Bi-grams outperform unigrams** — word pairs like "stock rises" capture sentiment better than individual words
2. **TF-IDF > Raw Counts** — weighting by importance improved model performance
3. **Simpler models can win** — Naive Bayes (96.65%) outperformed Random Forest (96.28%) while being faster to train
4. **News headlines carry strong predictive signal** for stock direction in this curated dataset

---


## 🤝 Connect

If you found this project helpful, give it a ⭐ on GitHub!

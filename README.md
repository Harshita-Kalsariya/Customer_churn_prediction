# 📉 P3 – Customer Churn (Compare Models + Imbalance & Tuning)

Customer churn deployment :->  https://customer-churn-prediction-rwtq.onrender.com
Telco customers ka churn predict karne wala AI/ML project. Part 1 me 4 models ko cross-validation se compare kiya, Part 2 me imbalance handle karke aur GridSearchCV se tuning ki.

## Folder Structure
```
P3_customer_churn/
├── frontend/   index.html, style.css
├── backend/    app.py
├── notebook/   churn_analysis.ipynb
├── datasets/   WA_Fn-UseC_-Telco-Customer-Churn.csv (Kaggle)
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset
Kaggle: **Telco Customer Churn** – https://www.kaggle.com/datasets/blastchar/telco-customer-churn
CSV ko `datasets/` folder me rakho.

## Setup
```bash
pip install -r requirements.txt
jupyter notebook notebook/churn_analysis.ipynb   # analysis
python backend/app.py                            # web app -> http://127.0.0.1:5000
```

## Checklist
- [x] 4 models trained (LogReg, Decision Tree, Random Forest, Gradient Boosting)
- [x] Comparison table made (5-fold CV)
- [x] Winner chosen with reason
- [x] Recall improved (class_weight)
- [x] Tuned with GridSearchCV
- [x] Plain-English finding written

## Key Finding
Month-to-month contract wale, kam tenure wale, fiber optic aur high monthly bill wale customers sabse zyada churn karte hain. Long-term contract aur tech support wale customers rukte hain.

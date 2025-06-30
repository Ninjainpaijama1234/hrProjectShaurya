# HR Attrition Insights Dashboard

A Streamlit web-app delivering enterprise-grade analytics on employee attrition
using the *EA.csv* dataset (1,470 employee records, 34 features).

## 🎯  Key Features
| Area | Highlights |
|------|------------|
| Macro View | High-level attrition by department, role, age, tenure, pay |
| Micro View | Drill-downs by gender, marital status, overtime, commute, satisfaction |
| Predictive | Random-Forest feature-importance & ROC-AUC indicator |
| Data Ops | CSV download, global sidebar filters, responsive layout |
| Tech | Streamlit, Plotly, scikit-learn, Pandas |

## 🏗 Folder Structure
├── app.py
├── EA.csv
├── requirements.txt
└── README.md

## 🚀  Local Run
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

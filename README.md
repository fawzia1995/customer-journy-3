
# Customer Journey Analysis System

**Project by:**  
- Habib Ali Badour – habib_294761 – C3  
- Fawzia Anwar Alomari – fwziea_284905 – C3  
- Anas Haider – Anas_308054 – C2  

**Course:** Machine Learning Techniques (MLT)  
**Degree:** Master of Computer Science (MCS)  
**Institution:** Ministry of Higher Education & Scientific Research, Syrian Arab Republic  

---

## 📌 Introduction

The **Customer Journey Analysis System** is designed to analyze customer interactions and improve decision-making in commercial opportunities.  
It focuses on understanding the sequence of interactions (Customer Journey) between clients and the company and predicting outcomes using **Decision Tree** models in Python.  

The system provides **Next Best Actions (NBA)** recommendations based on historical data, customer location (Country), and type of solution (Solution). It adapts dynamically as new interactions occur.

---

## 📝 Project Overview

- Analyze customer interactions chronologically to identify the most effective paths.  
- Extract top 5 interaction paths per group (by Country and Solution).  
- Use Decision Tree algorithm to evaluate the importance of each action.  
- Recommend the **top 4 next actions** in various contexts:  
  - By Country  
  - By Solution  
  - By Country & Solution combined  
- Dynamically update recommendations as new interactions are added.  
- Visualize optimal paths leading to won or lost opportunities using Sankey Diagrams.  

---

## 💻 Technologies Used

- **Programming Language:** Python  
- **Framework:** Streamlit (for interactive UI)  
- **Libraries:** Pandas, NumPy, scikit-learn, Matplotlib / Plotly (for Sankey diagrams)  

---

## ⚙️ Features

- Data cleaning and preprocessing  
- Customer account aggregation and journey path building  
- Top 5 interaction path extraction  
- Decision Tree analysis for action importance  
- Next Best Actions recommendation system  
- Dynamic weight adjustment for actions  
- Sankey Diagram visualization of optimal paths  

---

## 🛠️ Installation

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/customer-journey-system.git
cd customer-journey-system
```

2. Create a virtual environment and activate it:  
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. Install dependencies:  
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:  
```bash
streamlit run app.py
```

---

## 🗂️ Usage

- Upload your Excel dataset of customer interactions.  
- Explore Top Paths, Decision Tree analysis, and Next Best Actions recommendations.  
- Visualize optimal journeys with interactive Sankey Diagrams.  
- Add new interactions to dynamically update recommendations.  

---

## 📊 Outcome

- Improved understanding of customer behavior  
- Data-driven recommendations for sales strategies  
- Visualization of most effective interaction paths  
- Enhanced decision-making based on historical data  

---

## 📝 References

- Decision Tree algorithm for predictive analytics  
- Customer Journey analytics concepts  

---

## 📌 Conclusion

This system demonstrates how **Machine Learning** and **Data Analysis** can be applied to improve customer relationship management and predict commercial opportunity outcomes.  

It allows organizations to make informed decisions and optimize interaction strategies for better results.

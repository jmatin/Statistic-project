# 📊 Principal Component Analysis (PCA) on State Expenditure Data

## Overview

This project implements a **Principal Component Analysis (PCA)** on real-world state expenditure data, analyzing the distribution of public spending across 11 sectors over 24 consecutive years. The goal is to identify the main axes of variation in public expenditure and provide meaningful economic interpretations.

## 📁 Project Structure

```
├── doc/
│ ├── DataDepenses (1).csv # Raw expenditure dataset
│ ├── Devoir_Maison_ULB.pdf # Assignment description
├── src/
│ ├── ACP_en_python.py # Main PCA implementation in Python
│ ├── Devoir_Maison_Question5.pdf # Answer question 5
│ ├── Devoir_maison_stat2.pdf # Homework answers
│ ├── report_acp.pdf # Full PCA analysis report
└── README.md

```

## 📈 Expenditure Sectors Analyzed

| Abbreviation | Sector |
|:---:|---|
| PVP | Public Powers |
| AGR | Agriculture |
| CMI | Commerce & Industry |
| TRA | Labor |
| LOG | Housing & Land Development |
| EDU | Education |
| ACS | Social Action |
| ACO | War Veterans |
| DEF | Defense |
| DET | Debt |
| DIV | Miscellaneous |

## 🔬 Methodology

1. **Exploratory Data Analysis**
   - Descriptive statistics (mean, std, range, quartiles)
   - Covariance and correlation matrix analysis

2. **Data Standardization**
   - Variables standardized (centered and scaled) due to heterogeneous scales (notably DEF and DET)

3. **PCA Computation**
   - Eigenvalue decomposition of the correlation matrix
   - Scree plot analysis

4. **Axis Selection**
   - **Kaiser criterion**: 3 components with eigenvalue > 1
   - **Cumulative variance**: 75.57% explained with 3 axes
   - **Elbow rule**: clear break after 3rd component

5. **Interpretation**
   - Correlation analysis between original variables and principal components
   - Biplot visualization (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3)

## 📊 Key Results

| Axis | Variance Explained | Interpretation |
|:---:|:---:|---|
| PC1 | 45.21% | Balance between social/economic spending vs military/debt |
| PC2 | 18.63% | Short-term cyclical fluctuations (geopolitical context) |
| PC3 | 11.72% | Internal trade-offs within social spending sectors |

### Main Findings

- **PC1** reveals a structural transition over decades: from a budget dominated by debt and defense to increasing investments in education, social action, and agriculture
- **PC2** captures ~5-year cyclical patterns, likely linked to geopolitical events and veteran-related expenditure
- **PC3** highlights subtle trade-offs between different social spending categories

## 🛠️ Tech Stack

| Tool | Usage |
|---|---|
| Python | Main implementation |
| NumPy | Matrix computations |
| Pandas | Data manipulation |
| Matplotlib | Visualization |
| Scikit-learn | PCA computation |
| LaTeX | Report writing |

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/JoelMatin/PCA-State-Expenditure.git

# Navigate to source folder
cd src

# Run the PCA analysis
python ACP_en_python.py

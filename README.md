## 📑 **Task 1: Credit Scoring Business Understanding**

### 1️⃣ **How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?**

The **Basel II Accord** emphasizes risk-sensitive requirements for banks, particularly focusing on how credit risk is measured and managed. It requires financial institutions to have reliable, transparent, and auditable credit risk models to calculate capital requirements.

For this reason:

- The models must be **interpretable**, not just accurate, because auditors and regulators need to understand how risk scores are derived.
- A **well-documented model** ensures transparency, accountability, and compliance with regulatory standards.
- Models that are too complex (like black-box models) without proper explainability may not be accepted in high-stakes financial environments.

---

### 2️⃣ **Why is creating a proxy variable necessary, and what are the business risks of making predictions based on this proxy?**

- In this project, there is no direct label indicating whether a customer defaulted (i.e., failed to repay a loan).
- Therefore, we must create a **proxy target**, such as identifying disengaged customers (e.g., low frequency, low monetary value) as a signal for high risk.

However, using a proxy introduces risks:

- **Mislabeling:** The proxy might incorrectly classify customers who are simply inactive, not necessarily high risk.
- **Model bias:** If the proxy doesn't perfectly represent true defaults, the model may be biased or inaccurate.
- **Business impact:** Poor predictions could lead to either extending credit to risky customers (losses) or rejecting good customers (lost revenue).

---

### 3️⃣ **What are the key trade-offs between using a simple, interpretable model (e.g., Logistic Regression with WoE) vs. a complex, high-performance model (e.g., Gradient Boosting) in a regulated financial context?**

| Simple Models (e.g., Logistic Regression with WoE)                        | Complex Models (e.g., Gradient Boosting)                          |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| ✅ Highly interpretable — easy to explain to regulators and stakeholders. | 🚀 Higher predictive performance, especially on complex patterns. |
| ✅ Easy to audit, monitor, and deploy in compliance-driven settings.      | ❌ Harder to interpret; often considered a “black box.”           |
| 🔍 Assumptions are clear (e.g., linearity, monotonicity).                 | 🧠 Can model non-linear relationships and complex interactions.   |
| ⚙️ Fast to train, lightweight in production.                              | 💻 Requires more compute resources; complex deployment.           |
| 🛑 May have lower accuracy if relationships are not linear.               | ⚠️ Higher regulatory scrutiny if explainability is lacking.       |

**Conclusion:** In regulated environments like banking, there is often a preference for simpler, interpretable models unless explainability techniques (like SHAP) are applied to complex models.

### ✅ Task 3 Completed

- Built a robust feature engineering pipeline with:

  - Date extraction
  - Aggregation (customer-level)
  - Handling missing values
  - Scaling numeric features
  - Encoding categorical features

- Saved and tested the pipeline for future predictions.

### 🚀 Next Step:

- Task 4: Create a proxy target variable using RFM analysis and K-Means clustering.

### ✅ Task 4 Completed

- Computed Recency, Frequency, and Monetary (RFM) features.
- Applied KMeans clustering to segment customers into behavioral groups.
- Saved 'Cluster' as a proxy target for supervised modeling tasks.
- Ready to use this proxy target for the next step: model training.

### 🚀 Next Step:

- Task 5: Model Development using this proxy target.

### ✅ Task 5 Completed

- Built a machine learning pipeline to predict customer clusters.
- Achieved an accuracy of 99%.
- Model saved for future predictions.
- Ready for deployment or further analysis.

✅ Task 6 Full Breakdown
🏗️ Step 📄 Description
✅ API Development Create REST API using FastAPI
✅ Model Serving Load model from MLflow registry
✅ Prediction Endpoint /predict to serve risk predictions
✅ Data Validation Use Pydantic models for input/output
✅ Containerization Dockerfile + docker-compose.yml
✅ Continuous Integration GitHub Actions with flake8 + pytest

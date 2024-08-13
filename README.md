**Credit Card Default Prediction and Data Analysis, case of American Express Dataset**

**Introduction**:
Credit risk management is a cornerstone of financial stability for institutions like American Express, where the ability to accurately predict defaults is critical. Poor credit risk management can lead to significant financial losses, eroding profitability and undermining customer trust. This project addresses these challenges by leveraging advanced machine learning techniques to predict credit card defaults.

The analysis is crucial for several reasons:

**Mitigating Financial Losses:** Accurate predictions enable preemptive actions that reduce the risk of defaults, safeguarding the institution's financial health.

**Optimizing the Credit Portfolio:** Understanding the drivers of default helps in tailoring credit offerings, minimizing risk while maximizing returns.

**Regulatory Compliance:** A robust predictive model helps American Express meet regulatory requirements, ensuring the institution's operations are in line with financial regulations.

**Enhancing Customer Relations:** By predicting defaults, the institution can engage with at-risk customers proactively, offering solutions to prevent defaults and enhance customer loyalty.

**Strategic Decision-Making:** Insights from the analysis inform broader strategic decisions, from setting credit policies to managing capital reserves.


**Objective**:
Credit risk management is a cornerstone of financial stability for institutions like American Express, where the ability to accurately predict defaults is critical. Poor credit risk management can lead to significant financial losses, eroding profitability and undermining customer trust. This project addresses these challenges by leveraging advanced machine learning techniques to predict credit card defaults. This project aims to develop, compare, and validate multiple machine learning models to identify the most effective approach for accurately assessing credit risk. The focus is not only on achieving high predictive accuracy but also on understanding the underlying factors that contribute to borrower defaults. By identifying these factors, financial institutions can make informed decisions to mitigate risk, optimize credit offerings, and implement proactive risk management strategies. The study includes data preprocessing, model training, evaluation using multiple metrics, and comparison of model performance. The primary goal is to predict the likelihood of default for potential borrowers, with a detailed analysis of model performance metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

**Steps:**

**Data Preprocessing:** The American Express dataset was meticulously prepared by addressing missing values, encoding categorical variables, and standardizing numerical features. Stratified sampling was employed to ensure a balanced representation of defaulters and non-defaulters in the training and test sets.

**Exploratory Data Analysis (EDA):** Comprehensive EDA was conducted, involving the creation of visualizations like pie charts, box plots, and bar charts. These visualizations were essential for understanding the distribution of key variables and their relationship with credit defaults, helping to identify potential risk factors and data anomalies.

**Model Training and Evaluation:** Several machine learning models—Logistic Regression, Decision Trees, Random Forest, K-Nearest Neighbors (KNN), and XGBoost—were trained on the processed data. Each model’s performance was rigorously evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

**Model Comparison:** The performance of each model was systematically compared using ROC curves and a summary table of evaluation metrics to identify the most accurate and reliable model for predicting credit risk.

**Actionable Insights and Recommendations:** Based on the results from EDA and model evaluations, actionable insights were generated. These insights are intended to inform American Express's credit risk management strategies, including refining credit scoring systems, implementing risk-based pricing, and developing targeted financial products for different customer segments.

**Set of Analysis Performed:**
**Exploratory Data Analysis (EDA):** Conducted to identify trends, correlations, and outliers that could impact model performance.

**Feature Importance Analysis:** Performed using ensemble models like Random Forest and XGBoost to determine the most influential features in predicting defaults.

**Performance Metrics Evaluation:** Calculated metrics such as accuracy, precision, recall, F1-score, and ROC AUC to assess model performance.

**Model Comparison:** ROC curves were plotted to visually compare the models' ability to differentiate between defaulters and non-defaulters.

**Tools and Technologies Used:**
**Python Programming Language:** Python was used for all aspects of the project, including data processing, analysis, and modeling.
**Libraries:**
Pandas: For data manipulation and analysis.
NumPy: For numerical computations.
Scikit-Learn: For implementing machine learning algorithms, model evaluation, and data preprocessing.
Seaborn and Matplotlib: For data visualization and generating charts.
XGBoost: For training the XGBoost model, known for its performance on structured data.
LightGBM (if used): For training the LightGBM model, another powerful gradient boosting framework.

**Machine Learning Algorithms Used:**

**Logistic Regression:** A baseline linear model used for binary classification and understanding the impact of features on the likelihood of default.
**Decision Tree:** A non-linear model that splits data into branches to make predictions based on feature values.
**Random Forest:** An ensemble learning method that builds multiple decision trees and aggregates their predictions to improve accuracy and robustness.
**K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies instances based on the majority class among the nearest neighbors.
**XGBoost:** A gradient boosting algorithm that builds an ensemble of weak learners (decision trees) to improve prediction accuracy.
**Results:**
**Best Performing Model:** XGBoost and Random Forest were identified as the best-performing models, with the highest accuracy and ROC AUC scores.
Key Features: Credit score, credit limit used, and net yearly income were identified as key factors influencing the likelihood of default.
Model Interpretability: While XGBoost provided the highest accuracy, Logistic Regression offered valuable insights into the relationship between features and default risk.
Insights for American Express: The analysis provided actionable recommendations for improving credit risk management, including risk-based pricing, credit limit adjustments, and targeted financial products for different customer segments.

**IMPLEMENTATION:**
**Data Preprocessing:**

**Data Loading:** The datasets for training and testing were loaded and combined to ensure consistent encoding of categorical variables.

**Handling Missing Values:** The datasets were checked for missing values, and missing values in the target variable were filled with the mode.

**Stratified Sampling**: Stratified sampling was performed to ensure that both classes are represented in the training and test sets.

**Feature Scaling**:Feature scaling was applied to the datasets to ensure that all features contribute equally to the model training.

**Model Performance Comparison:**
**ROC Curve**:

The ROC curves for the various models are plotted to visualize the trade-off between true positive rate and false positive rate.

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/ROC%20curve.jpg" alt="Description" width="300"/>

Summary Table
The performance metrics for the various models are summarized in a table.

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Summary%20Table.jpg" alt="Description" width="300"/>

 
**Analysis and Conclusion:**

Analysis
The performance of various machine learning models was evaluated based on accuracy and ROC AUC scores. The ROC curves provide a visual representation of the trade-off between true positive rate and false positive rate for each model

**Actionable Insights and Recommendations:**
Based on the comprehensive analysis and model evaluation, the following actionable insights and recommendations are provided to enhance credit risk management and predictive modeling strategies:

**1. Model Selection for Credit Risk Prediction**
      •	**XGBoost and Random Forest models** have demonstrated the highest accuracy and ROC AUC scores, making them the most reliable models for predicting credit risk in this dataset.
      •	These models should be prioritized for deployment in production environments due to their superior performance in distinguishing between defaulters and non-defaulters.
      •	**Logistic Regression**, while less accurate than ensemble methods, can still be a valuable tool for interpretability and transparency, especially in regulatory environments where model explainability is critical.

**2. Data Handling and Preprocessing**
      • **Stratified Sampling:** Ensuring that the training and test sets are representative of the full dataset is crucial. The use of stratified sampling in this analysis prevented class imbalance issues and ensured 
          robust model evaluation.
      •	**Feature Scaling and Imputation:** These preprocessing steps should be standard practice in any credit risk modeling pipeline. StandardScaler effectively normalized the data, and SimpleImputer handled missing 
        values, which is vital for maintaining the integrity of the model inputs.
        
**3. Model Interpretability**
      •	While complex models like XGBoost and Random Forest offer higher accuracy, they can be challenging to interpret. For cases where model transparency is necessary, such as in financial auditing or regulatory 
        compliance, simpler models like Logistic Regression should be considered.
      •	Feature Importance Analysis using Random Forest or XGBoost can provide insights into the most critical factors affecting credit risk, which can be valuable for refining credit policies.
      
**4. Ensemble Methods**
      •	The use of ensemble methods (Random Forest, XGBoost) has shown significant benefits in improving prediction accuracy. Financial institutions should consider implementing ensemble techniques as part of their 
        predictive analytics toolkit.
      •	Blending Models: Consider blending multiple models (e.g., combining predictions from Random Forest, XGBoost, and Logistic Regression) to achieve even more robust predictions.
      
**5. Continuous Model Monitoring and Updating**
      •	The credit risk environment is dynamic, with new data and patterns emerging regularly. Implementing a system for continuous model monitoring and periodic updates will ensure that the models remain accurate and 
        relevant over time.
      •	Performance Tracking: Regularly track model performance metrics (accuracy, ROC AUC) and recalibrate models when significant drops in performance are detected.
      
**6. Deployment and Scalability**
     •	Prioritize the deployment of models that not only perform well but also scale efficiently with large datasets. XGBoost, known for its scalability, should be considered for deployment in systems handling high 
       volumes of credit applications.
     •	Ensure that the model deployment pipeline includes capabilities for real-time scoring, especially for online credit applications where quick decisions are crucial.
     
**7. Regulatory and Ethical Considerations**
     •	Ensure that all models, especially black-box models like XGBoost, are subjected to thorough testing for fairness and bias. It is essential to ensure that credit decisions do not inadvertently discriminate against 
       protected groups.
     •	Transparency: Develop strategies for explaining model decisions to both internal stakeholders and regulators. For example, use Logistic Regression as a baseline for interpretability, and complement it with the 
       accuracy of ensemble models.
       
**8. Exploring Additional Data Sources**
     •	To further improve model accuracy and predictive power, consider integrating additional data sources such as transactional data, social media data, or alternative credit scores. These can provide richer insights 
       into borrower behavior.
     •	Alternative Data: For thin-file or no-file customers (those with limited credit history), exploring alternative data sources can help in building more inclusive credit risk models.
      
**9. Customer Segmentation**
    •	Use model insights to segment customers based on risk levels. This segmentation can drive more targeted marketing efforts, personalized loan products, and differentiated interest rates based on risk profiles.
    •	High-Risk Customer Management: Develop specific strategies for managing high-risk customers, such as offering smaller loans with higher interest rates or requiring additional collateral.
**10. Future Research and Model Enhancement**
    •	Hyperparameter Tuning: While the models used in this study provided strong results, further tuning of hyperparameters, especially in XGBoost and Random Forest, could yield even better performance.
    •	Model Explainability: Invest in research and development of explainable AI (XAI) methods to make complex models more transparent, which is increasingly becoming a regulatory requirement in finance.
    •	Deep Learning Approaches: Explore the potential of deep learning models, such as neural networks, which might capture complex patterns in the data that traditional models miss.
By implementing these recommendations, financial institutions can significantly enhance their credit risk management practices, leading to more accurate lending decisions, reduced default rates, and improved overall financial stability.




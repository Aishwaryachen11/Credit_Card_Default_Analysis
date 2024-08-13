# **CREDIT CARD DEFAULT PREDICTION AND DATA ANALYSIS, CASE OF AMAERICAN EXPRESS DATASET**

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

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/ROC%20curve.jpg" alt="Description" width="600"/>

Summary Table
The performance metrics for the various models are summarized in a table.

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Summary%20Table.jpg" alt="Description" width="450"/>

 
**Analysis and Conclusion:**

Analysis
The performance of various machine learning models was evaluated based on accuracy and ROC AUC scores. The ROC curves provide a visual representation of the trade-off between true positive rate and false positive rate for each model

**Actionable Insights and Recommendations:**
Based on the comprehensive analysis and model evaluation, the following actionable insights and recommendations are provided to enhance credit risk management and predictive modeling strategies:

**1. Model Selection for Credit Risk Prediction**
      	**XGBoost and Random Forest models** have demonstrated the highest accuracy and ROC AUC scores, making them the most reliable models for predicting credit risk in this dataset.
      	These models should be prioritized for deployment in production environments due to their superior performance in distinguishing between defaulters and non-defaulters.
      	**Logistic Regression**, while less accurate than ensemble methods, can still be a valuable tool for interpretability and transparency, especially in regulatory environments where model explainability is critical.

**2. Data Handling and Preprocessing**
       **Stratified Sampling:** Ensuring that the training and test sets are representative of the full dataset is crucial. The use of stratified sampling in this analysis prevented class imbalance issues and ensured 
          robust model evaluation.
      	**Feature Scaling and Imputation:** These preprocessing steps should be standard practice in any credit risk modeling pipeline. StandardScaler effectively normalized the data, and SimpleImputer handled missing 
        values, which is vital for maintaining the integrity of the model inputs.
        
**3. Model Interpretability**
      	While complex models like XGBoost and Random Forest offer higher accuracy, they can be challenging to interpret. For cases where model transparency is necessary, such as in financial auditing or regulatory 
        compliance, simpler models like Logistic Regression should be considered.
      	Feature Importance Analysis using Random Forest or XGBoost can provide insights into the most critical factors affecting credit risk, which can be valuable for refining credit policies.
      
**4. Ensemble Methods**
      	The use of ensemble methods (Random Forest, XGBoost) has shown significant benefits in improving prediction accuracy. Financial institutions should consider implementing ensemble techniques as part of their 
        predictive analytics toolkit.
      	Blending Models: Consider blending multiple models (e.g., combining predictions from Random Forest, XGBoost, and Logistic Regression) to achieve even more robust predictions.
      
**5. Continuous Model Monitoring and Updating**
      	The credit risk environment is dynamic, with new data and patterns emerging regularly. Implementing a system for continuous model monitoring and periodic updates will ensure that the models remain accurate and 
        relevant over time.
      	Performance Tracking: Regularly track model performance metrics (accuracy, ROC AUC) and recalibrate models when significant drops in performance are detected.
      
**6. Deployment and Scalability**
     	Prioritize the deployment of models that not only perform well but also scale efficiently with large datasets. XGBoost, known for its scalability, should be considered for deployment in systems handling high 
       volumes of credit applications.
     	Ensure that the model deployment pipeline includes capabilities for real-time scoring, especially for online credit applications where quick decisions are crucial.
     
**7. Regulatory and Ethical Considerations**
     	Ensure that all models, especially black-box models like XGBoost, are subjected to thorough testing for fairness and bias. It is essential to ensure that credit decisions do not inadvertently discriminate against 
       protected groups.
     	Transparency: Develop strategies for explaining model decisions to both internal stakeholders and regulators. For example, use Logistic Regression as a baseline for interpretability, and complement it with the 
       accuracy of ensemble models.
       
**8. Exploring Additional Data Sources**
     	To further improve model accuracy and predictive power, consider integrating additional data sources such as transactional data, social media data, or alternative credit scores. These can provide richer insights 
       into borrower behavior.
     	Alternative Data: For thin-file or no-file customers (those with limited credit history), exploring alternative data sources can help in building more inclusive credit risk models.
      
**9. Customer Segmentation**
    	Use model insights to segment customers based on risk levels. This segmentation can drive more targeted marketing efforts, personalized loan products, and differentiated interest rates based on risk profiles.
    	High-Risk Customer Management: Develop specific strategies for managing high-risk customers, such as offering smaller loans with higher interest rates or requiring additional collateral.
     
**10. Future Research and Model Enhancement**
    	Hyperparameter Tuning: While the models used in this study provided strong results, further tuning of hyperparameters, especially in XGBoost and Random Forest, could yield even better performance.
    	Model Explainability: Invest in research and development of explainable AI (XAI) methods to make complex models more transparent, which is increasingly becoming a regulatory requirement in finance.
    	Deep Learning Approaches: Explore the potential of deep learning models, such as neural networks, which might capture complex patterns in the data that traditional models miss.
By implementing these recommendations, financial institutions can significantly enhance their credit risk management practices, leading to more accurate lending decisions, reduced default rates, and improved overall financial stability.

**EXPLORATORY DATA ANALYSIS**

Actionable Insights, Trends, and Strategic Recommendations from EDA:

**1. Defaulter Ratio (Pie Chart)**

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Defaulter%20Ratio.jpg" alt="Description" width="450"/>
 
•	**Insight:** The defaulter ratio indicates that only 8.2% of the borrowers defaulted on their credit cards, while 91.8% did not default. This shows a significant class imbalance in the dataset. A low defaulter rate is observed, which is common in credit risk datasets as most customers generally repay their debts.

•	**Recommendation:**
Modeling Strategy: Given the class imbalance, consider using techniques such as SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class (defaulters) or use algorithms like XGBoost that handle class imbalance well.
Risk Management: Although the defaulter rate is low, financial institutions should still focus on improving prediction accuracy for the minority class to avoid potential financial losses.

**2. Age Distribution (Density Plot)**

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Age%20Distribution.jpg" alt="Description" width="450"/>
 
•	**Insight:** The age distribution of the borrowers is fairly uniform across the age ranges of 20 to 55, with noticeable peaks around the ages of 30 and 55. Borrowers in their early 30s and mid-50s represent significant portions of the dataset, potentially indicating life stages associated with higher financial activity (e.g., buying homes, retirement planning).

•	**Recommendation:**
Targeted Marketing: Financial institutions should tailor their product offerings to these age groups, considering their financial needs. For example, younger borrowers may benefit from lower interest rates on loans, while older borrowers may be interested in retirement planning products.
Risk Profiling: Age can be a factor in credit risk profiling. Institutions should assess the correlation between age and default risk to refine their credit scoring models.

**3. Credit Score vs. Credit Card Default (Box Plot)**

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Credit%20Score%20Vs%20Credit%20Card%20Default.jpg" alt="Description" width="450"/>
 
•	**Insight:** The box plot shows that individuals who defaulted have significantly lower credit scores (around 600) compared to those who did not default (around 800). A clear inverse relationship between credit score and default risk is observed, with lower credit scores being associated with a higher likelihood of default.

•	**Recommendation:**
Credit Scoring Systems: Enhance credit scoring models by integrating additional behavioral and transactional data to predict defaults more accurately.
Risk-Based Pricing: Implement risk-based pricing strategies where borrowers with lower credit scores are charged higher interest rates or required to provide additional collateral.

**4. Credit Limit Used vs. Credit Card Default (Box Plot)**

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Credit%20limit%20Used%20Vs%20Credit%20Card%20Default.jpg" alt="Description" width="450"/>
 
•	**Insight:** Defaulters typically use a much higher percentage of their credit limit (around 80-90%) compared to non-defaulters (around 40-60%). High utilization of credit limits is a strong indicator of financial distress and increased default risk.

•	**Recommendation:**
o	Credit Limit Management: Financial institutions should monitor customers with high credit utilization closely and consider proactive measures such as offering financial counseling or adjusting credit limits.
o	Early Warning Systems: Develop early warning systems that flag customers who consistently use a high percentage of their credit limit, allowing for pre-emptive actions to mitigate default risk.

**5. Net Monthly Income by Occupation Type (Bar Chart)**

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Net%20Monthly%20Income%20by%20Occupation%20Type.jpg" alt="Description" width="600"/>
 
•	**Insight:** Income levels vary significantly across different occupation types, with certain occupations like Managers and IT Staff earning higher than others, such as Laborers and Cleaning Staff. Higher-income occupations generally show lower default rates, while lower-income occupations may have higher default rates.

•	**Recommendation:**
o	Customized Credit Products: Offer tailored credit products that align with the income levels of different occupations. For example, lower-income groups could benefit from microloans with manageable repayment terms.
o	Income Verification: Ensure robust income verification processes are in place, especially for high-risk occupations, to accurately assess borrowers' repayment capacity.

**6. Gender Distribution and Default (Bar Chart)**

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Gender%20Distribution%20Vs%20Default.jpg" alt="Description" width="450"/>
 
•	**Insight:** The majority of the dataset comprises female borrowers, with both genders showing a relatively low default rate. However, a slight imbalance is observed with more female borrowers represented. Gender distribution suggests that both male and female borrowers are equally prone to default, with no significant gender-based bias in default rates.

•	**Recommendation:**
o	Gender-Neutral Policies: Maintain gender-neutral credit policies, ensuring that creditworthiness assessments are based purely on financial indicators rather than demographic factors.
o	Diversity in Lending: Consider strategies to ensure that lending products are equally accessible to all genders, promoting financial inclusion across diverse demographics.

**Conclusion**
The exploratory data analysis provides valuable insights into the patterns and factors influencing credit card defaults. By leveraging these insights, financial institutions can implement targeted strategies to manage credit risk more effectively, enhance customer satisfaction, and maintain financial stability. The combination of data-driven insights and strategic recommendations forms the foundation for a robust credit risk management framework.

**Additional Insights and Recommendations Based on the Box Plots**

**1. Net Yearly Income, No. of Days Employed, Yearly Debt Payments (Box Plots)**

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Box%20Plot%201.jpg" alt="Description" width="550"/>
 
•	**Insight:**
o	Net Yearly Income: The net yearly income box plot shows a large number of outliers on the higher end, indicating that while most individuals have moderate incomes, a few individuals earn significantly more. The majority of the data is concentrated at the lower end of the income spectrum.
o	No. of Days Employed: The box plot for the number of days employed shows a significant outlier far above the rest of the data points, indicating that one or a few individuals have been employed for an unusually long period, potentially due to incorrect data entry.
o	Yearly Debt Payments: Similar to income, yearly debt payments also show a concentration of data at the lower end, with several outliers indicating higher debt payments.
•	Trend:
o	There is a significant disparity in income and debt payments, with most individuals earning and paying within a certain range, but a few outliers represent very high values.
o	The extreme outlier in the "No. of Days Employed" may suggest a data quality issue that needs further investigation.

•	**Recommendation:**
o	Outlier Management: Implement robust data cleaning processes to investigate and address potential data entry errors, especially the extreme outlier in the "No. of Days Employed" variable.
o	Targeted Lending: Financial institutions should consider offering differentiated financial products to individuals in the higher-income bracket, who may have different financial needs compared to the majority of borrowers.
o	Debt Repayment Strategies: Develop personalized debt repayment plans for borrowers with higher yearly debt payments to mitigate default risk.

**2. Credit Limit and Credit Score (Box Plots)**

<img src="https://github.com/Aishwaryachen11/Credit_Card_Default_Analysis/blob/main/Images/Box%20Pot%202.jpg" alt="Description" width="550"/>
 
•	**Insight:**
Credit Limit: The credit limit box plot reveals that most borrowers have relatively low credit limits, but there are outliers with significantly higher limits. This could indicate that a small portion of the population is being offered much higher credit limits, possibly due to higher income or better creditworthiness.
Credit Score: The box plot for credit scores shows a wide range, with most borrowers falling between a credit score of 600 and 900. A higher credit score is generally indicative of better creditworthiness.
•	Trend:
Higher credit limits are granted to a select few, likely those with higher income or better credit histories.
Credit scores vary widely among borrowers, highlighting the diverse credit profiles within the dataset.

•	**Recommendation:**
Credit Limit Adjustment: Regularly reassess and adjust credit limits based on changes in borrowers’ financial circumstances and credit behavior. This will help in managing risk and maintaining healthy credit exposure.
Credit Scoring Enhancements: Consider using advanced credit scoring models that incorporate more dynamic factors such as recent financial behavior and broader financial data to provide a more accurate assessment of creditworthiness.
Customer Education: Financial institutions should provide credit education programs to help customers understand the factors that affect their credit score and how to improve it, which can lead to better credit management and reduced default risk.

**Conclusion**
The box plots provide a deeper understanding of the distribution of key financial variables within the dataset, highlighting outliers and trends that could impact credit risk management strategies. By implementing the recommendations derived from these insights, financial institutions can better manage risk, improve customer segmentation, and tailor financial products to meet the diverse needs of their customers.









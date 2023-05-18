# credit-card-fraud-detection-stamatics
Detecting credit card fraud using Logistic Regression on the '[Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)'.

### Details about the dataset:
> The dataset contains 1000000 data points.
>
> We are trying to predict if a transaction made was fraudulent or not.
>
> 'fraud' is our binary independent variable.
>
> Data has no missing values.
>
> Dataset contains only 9% fraudulent transactions, i.e., it as an imbalanced dataset.


### Outlier Detection
> We remove data points which lie outside of “mean ± 3*standard deviation".
> 
> The final dataset has 963,001 data points.

### Exploratory Data Analysis
<img width="1000" alt="image" src="https://github.com/akshatg20/credit-card-fraud-detection-stamatics/assets/84704822/4274ac24-0db6-4b10-a82b-cedbb5d0510a">

> ”Fraud” seems to have the highest correlation with ‘ratio_to_median_purchase_price’ , followed by ‘distance_from_home’ and ‘online_order’.
> This also applies in real life as any fraudulent transaction would have the of theft of money to be comparatively higher than regular transactions.

<img width="345" alt="image" src="https://github.com/akshatg20/credit-card-fraud-detection-stamatics/assets/84704822/d46045fa-7015-4d16-94cd-2715a9a52e4c"> <img width="340" alt="image" src="https://github.com/akshatg20/credit-card-fraud-detection-stamatics/assets/84704822/4e00413c-37df-4940-960c-8718cb2fcc52"> <img width="345" alt="image" src="https://github.com/akshatg20/credit-card-fraud-detection-stamatics/assets/84704822/67664708-93e9-4475-8052-1896bca8f2bc">

> 95.23 % of frauds occur when it is an online order.
> 
> The frauds occurred seem to have occurred not close from their homes.


### Multicollinearity Test

> A variance inflation factor(VIF) detects multicollinearity in regression analysis.
> 
> Multicollinearity is when there’s correlation between predictors  in a model; it’s presence can adversely affect your regression results.
> 
> The VIF estimates how much the variance of a regression coefficient is inflated due to multicollinearity in the model.
> 
> The VIF test shows us that transactions from the **repeat retailer** has a high VIF indicating that it is dependent on the other attributes present. Thus, we remove this feature from our model.


### Running Logistic Regression












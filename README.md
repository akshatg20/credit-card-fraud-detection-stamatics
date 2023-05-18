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
<img width="1000" alt="image" src="https://github.com/akshatg20/credit-card-fraud-detection-stamatics/blob/main/images/heatmap.png">

> ”Fraud” seems to have the highest correlation with ‘ratio_to_median_purchase_price’ , followed by ‘distance_from_home’ and ‘online_order’.
> This also applies in real life as any fraudulent transaction would have the of theft of money to be comparatively higher than regular transactions.

<img width="750" alt="image" src="https://github.com/akshatg20/credit-card-fraud-detection-stamatics/blob/main/images/eda.png"> 

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
``` bash
#Scaling features
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
X[['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price']] = min_max_scaler.fit_transform(X[['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price']])
X.describe()
```
#### Using K-means clustering
K-means clustering uses “centroids”, K different randomly-initiated points in the data, and assigns every data point to the nearest centroid.  After every point has been assigned, the centroid is moved to the average of all of the points assigned to it.  

We form 10 clusters and add that as a new feature in our logistic regression model.

``` bash
from sklearn.cluster import KMeans
features = ["distance_from_home","distance_from_last_transaction","ratio_to_median_purchase_price"]
kmeans = KMeans(n_clusters = 10,n_init=10,random_state=0)
X["Cluster"] = kmeans.fit_predict(X)
```
#### Results
> We import Logistic Regression from sklearn.linear_model and get an accuracy of ** 97.23%**/

<img width="786" alt="image" src="https://github.com/akshatg20/credit-card-fraud-detection-stamatics/assets/84704822/4045e2c4-dbd7-4f5f-be1f-f96eb44c2bdb">


#### Classification Report

<img width="801" alt="image" src="https://github.com/akshatg20/credit-card-fraud-detection-stamatics/assets/84704822/83305ba3-4a78-4d66-aca8-0f71cf20b338">

#### ROC Curve
> An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters: True Positive Rate. False Positive Rate. 
> 
> We achieve an AUC score of 0.86.

![ROC Curve](https://drive.google.com/file/d/1Xa8V3cetLvsj9n2XczuZ4RdraMaPS6yk/view?usp=share_link)

### SMOTE Oversampling
> SMOTE is an oversampling technique where the synthetic samples are generated for the minority class.

``` bash
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model.fit(X_res,y_res)
y_res_pred = model.predict(X_test)
```

> After applying oversampling, model is much better at reducing false negatives.
>
> Model has an impressive AUC score of 0.96.
> 
> Lower precision but much higher recall.


### Conclusion
> OUR LOGISTIC REGRESSION MODEL CAN PREDICT WITH A VERY HIGH ACCURACY WHETHER A TRANSACTION IS FRAUDULENT OR NOT.
> 
> WE CAN MINIMIZE THE CASES WHERE WE FALSELY PREDICT FRUADULENT TRANSACTIONS AS NOT FRAUD USING OVERSAMPLING.
> 
> THE ODDS OF FRUAD SEEM TO INCREASE WITH AN INCREASE IN FACTORS LIKE DISTANCE FROM HOME, DISTANCE FROM LAST TRANSACTION, AND THE RATIO TO MEDIAN PURCHASE.
> 
> MOST OF THE FRUADS OCCUR THROUGH ONLINE TRANSACTIONS.













import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from joblib import load, dump

#IMPORTING DATASET
housing= pd.read_csv('data_housing.csv')

#FINDING STATS ABOUT THE DATA
'''print(housing.head())
print(housing.info())
print(housing.CHAS)
print(housing.CHAS.value_counts())
print(housing.describe())'''


#VISUALISING THE DATA
'''housing.hist(bins=50, figsize=(20,15))
plt.tight_layout()
plt.show()'''

#SPLITING THE DATA USING STRATIFIED SHUFFLE SPLIT SO THAT ALL TYPES OF VALUE ARE THERE IN TRAINING DATA
split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing.CHAS):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


'''print(strat_train_set)
print(strat_train_set.describe())
print(strat_train_set.info())
print(strat_train_set.CHAS.value_counts())'''


#LOOKING FOR CORRELATIONS
corr_matrix= housing.corr()

print("\nCorrelation of attributes with our label MEDV")
print(corr_matrix.MEDV.sort_values(ascending=False))

#we will visualise correlation of these features with each other
attributes= ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12,8))

#i found garph between RM and MEDV to be interesting: so we will ignore outlier points/points which break trend
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)
plt.show()

#SEPARATING FEATURE AND MODEL FOR TRAINING SET 
housing=strat_train_set.drop("MEDV", axis=1)           #NOTE: WE HAVE DONE THIS STEP AFTER MISSING ATTRIBUTE STEP
housing_labels= strat_train_set["MEDV"].copy()

#SEPARATING TRAINING AND MODEL FOR TEST SET
housing_test=strat_test_set.drop("MEDV", axis=1)
housing_test_label=strat_test_set["MEDV"].copy()


#SOLVING MISSING ATTRIBUTES: I deleted 5 values from RM attribute of dataset
#we will replace missing datapoints with median value
median= housing["RM"].median()
'''housing["RM"].fillna(median, inplace=True)''' #we can do this but sklearn already has a class to do this: Imputer

imputer= SimpleImputer(strategy="median")
imputer.fit(housing)                          #finds median for each row

'''print(imputer.statistics_)
print(imputer.statistics_.shape)'''

#we craete a new dataframe, where missing values are replaced by median; we have automated it
X=imputer.transform(housing)
housing_tr=pd.DataFrame(X, columns=housing.columns)

'''print(housing_tr.describe())'''         #we see length of RM is back to 506 later 404 as we copied strat_train_set into housing
                                           #so housing_tr is now our training set


#CREATING A PIPELINE            
#NOTE: Pipeline should be created from the beginning of the project 
# and all the steps we did before would be done under it, in professional work
my_pipeline=Pipeline([('imputer', SimpleImputer(strategy="median")),
                      ('std_scaler', StandardScaler())])

housing_num_tr=my_pipeline.fit_transform(housing_tr)                         #it is a numpy array as predictors take array as input

#print(housing_num_tr)

#SELECTING A DESIRED MODEL
#1st TESTING LinearRegression() MODEL
model1= LinearRegression()
model1.fit(housing_num_tr, housing_labels)

#testing some predictions of 1st Model
'''some_data=housing.iloc[:5]
some_label= housing_labels.iloc[:5]

prepared_data= my_pipeline.transform(some_data)

pred=model1.predict(prepared_data)
print(pred)                            #[23.98694849 27.24843552 20.58163236 25.04097716 23.77039676]
print(list(some_label))'''             #[21.9, 24.5, 16.7, 23.1, 23.0]


#EVALUATING THE 1st MODEL
housing_predict= model1.predict(housing_num_tr)
lin_mse= mean_squared_error(housing_labels, housing_predict)
lin_rmse=np.sqrt(lin_mse)
#print(lin_rmse)                                  #4.8338850932262325      we should test another model


#2nd TESTING DecisionTreeRegression MODEL
model2= DecisionTreeRegressor()
model2.fit(housing_num_tr, housing_labels)

#testing some predictions of 2nd Model
'''some_data=housing.iloc[:5]
some_label= housing_labels.iloc[:5]

prepared_data= my_pipeline.transform(some_data)

pred=model2.predict(prepared_data)
print(pred)                              #[21.9 24.5 16.7 23.1 23. ]
print(list(some_label))'''               #[21.9, 24.5, 16.7, 23.1, 23.0]

#EVALUATING THE 2nd MODEL
housing_predict= model2.predict(housing_num_tr)
mse= mean_squared_error(housing_labels, housing_predict)
rmse=np.sqrt(mse)
#print(rmse)                        #0.0                this is the case of overfitting

#3rd TESTING RandomForestRegressor MODEL
model3= RandomForestRegressor()
model3.fit(housing_num_tr, housing_labels)

#testing some predictions of 3rd Model
'''some_data=housing.iloc[:5]
some_label= housing_labels.iloc[:5]

prepared_data= my_pipeline.transform(some_data)

pred=model3.predict(prepared_data)
print(pred)                                    #[22.324 25.016 16.686 23.361 23.598]
print(list(some_label))'''                     #[21.9, 24.5, 16.7, 23.1, 23.0]

#EVALUATING THE 3rd MODEL
housing_predict= model3.predict(housing_num_tr)
mse_r= mean_squared_error(housing_labels, housing_predict)
rmse_r=np.sqrt(mse_r)
#print(rmse_r)                                  #1.2869637420173592

#NOTE: REMEMBER DO NOT USE TEST DATA UNLESS YOU ARE SURE THE MODEL IS GOOD ONE

#USING BETTER EVALUATION TECHNIQUE: CROSS EVALUATION
#1st for LinearRegression
scores_lin= cross_val_score(model1, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores_lin= np.sqrt(-scores_lin)               #[4.22504643 4.26963547 5.08994166 3.83316228 5.37523674 4.4077251
                                                    #7.47616084 5.48181291 4.14392464 6.06190757]


#2nd for DecisionTreeRegression
scores_tree= cross_val_score(model2, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores_tree= np.sqrt(-scores_tree)          #[4.23942012 5.70791064 5.12761534 4.07445941 4.22152224 3.12921715
                                                 # 5.01128726 3.9727824  3.15190419 3.89348815]
                                                 #this rmse is better now, since it is not overfitted and also better than linear regression one


#3rd for RandomForestRegressor
scores_random= cross_val_score(model3, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores_random= np.sqrt(-scores_random)            #[2.88636975 2.81570923 4.36517147 2.61277215 3.45189214 2.69333983
                                                       #4.67553243 3.26803373 3.32402841 3.2289948 ]
                                                       #this one look even better model
                         

#NOTE: SO WE HAVE SELECTED: RandomForestRegressor() model, i.e: model3


#FUNCTION TO PRINT MEAN, STD DEVIATION, AND SCORES
def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Std. Deviation: ", scores.std())

print("RMSE scores for LinearRegression()")
print_scores(rmse_scores_lin)

print("\nRMSE scores for DecisionTreeRegression()")
print_scores(rmse_scores_tree)

print("\nRMSE scores for RandomForestRegressor()")
print_scores(rmse_scores_random)

#NOTE: Seeing mean of both also shows that RandomForestRegressor is better

#TO SAVE MODEL:
dump(model3, "Dragon.joblib")

#NOW TO USE TEST DATA:
housing_test_prepared=my_pipeline.transform(housing_test)
final_predictions= model3.predict(housing_test_prepared)
final_mse=mean_squared_error(housing_test_label, final_predictions)
final_rmse= np.sqrt(final_mse)

print("\nFinal predictions using Test Data:")
print(final_predictions)
print("\nActual labels of Test data:")
print(list(housing_test_label))

print(f"\nFinal RMSE of model using Test data: {final_rmse} \n\n")

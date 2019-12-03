import pandas as pd
import numpy as np

df_mlb=pd.read_csv("../data/mlb_elo.csv")
df_mlb_latest=pd.read_csv("../data/mlb_elo_latest.csv")


# ## MLB DataFrame



print(df_mlb.head(10))


# ## MBL Latest DataFrame
print(df_mlb_latest.head(10))

print(df_mlb.isna().score1.any())
print(df_mlb.isna().score2.any())

teams = df_mlb['team1']. unique()

avgAway=df_mlb.groupby('team2').score2.mean()
avgAwayList=[]
for x in range(0, len(avgAway)):
    avgAwayList.append(avgAway[x])


avgHome=df_mlb.groupby(['team1']).score1.mean()


avgHomeList=[]
for x in range(0, len(avgHome)):
    avgHomeList.append(avgHome[x])


import matplotlib.pyplot as plt
plt.scatter(avgHomeList, avgAwayList)


dict = {'Home': avgHomeList, 'Away': avgAwayList}
df_Scores = pd.DataFrame(dict)
df_Scores.head(10)


X = df_Scores['Home'].values.reshape(-1,1)
y = df_Scores['Away'].values.reshape(-1,1)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm



#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

y_pred = regressor.predict(X_test)
y_pred

y_pred = [y for x in y_pred for y in x]
y_test = [y for x in y_test for y in x]
X_test = [y for x in X_test for y in x]
X_train = [y for x in X_train for y in x]
y_train = [y for x in y_train for y in x]

dict = {'Actual': y_test, 'Predicted': y_pred}
df_pred = pd.DataFrame(dict)
df_pred

df_pred.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ## Linear Regression


plt.scatter(avgHome, avgAway,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel("Average Home Score")
plt.ylabel("Average Away Score")
plt.show()

list1=df_mlb['team1'].unique()
list1=list(list1)
list1=sorted(list1)

df = pd.DataFrame({'Home': avgHomeList, 'Away': avgAwayList}, index=list1)
fig,ax=plt.subplots()
plt.scatter(df['Home'], df['Away'])
plt.show()

for name,row in df.iterrows():
  plt.scatter(row['Home'],row['Away'], label=name)
plt.title("Average Away Scores vs Average Home Scores")
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Avg Home Score')
plt.ylabel('Avg Away Score')
plt.legend(title="Teams")
leg = plt.legend( loc = 'lower right', title="Teams")

plt.draw() # Draw the figure so you can find the positon of the legend.
# Get the bounding box of the original legend
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
# Change to location of the legend.
xOffset = .8
bb.x0 += xOffset
bb.x1 += xOffset
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
plt.show()
# ## Gradient Boosting Regression
from sklearn import ensemble
from sklearn import linear_model


plt.figure(figsize=(10, 5))
plt.title("Residuals $(y - \hat{y})$ of Linear Regression model")
arr=np.array(y_test)-np.array(y_pred)
#list1=y_test-y_pred
print(arr)
plt.scatter(X_test,arr, color='brown')
plt.show()

params = {
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 1,
    'criterion': 'mse'
}
x=np.array(X_test)
x=x.reshape(-1, 1)
y=np.array(y_pred)
y=y.reshape(-1, 1)
gradient_boosting_regressor = ensemble.GradientBoostingRegressor(**params)

gradient_boosting_regressor.fit(x, y)


plt.figure(figsize=(10, 5))
plt.title('Gradient Boosting model (1 estimators, Single tree split)')
plt.scatter(avgHome, avgAway,  color='black')
plt.plot(x, gradient_boosting_regressor.predict(x), color='r')
plt.show()


# The gradient boosting regression exhibits similar conclusions as the linear regression.

# # Conclusions:
# I used linear regression and gradient boosting to predict the average score for a home game and away game for each team. I found each team's average home and away scores and plotted them. I then used regression and gradient boosting to create the line of best fit for this plot. Both types of regression prove that a team scores better when they play at home rather than when they play away.

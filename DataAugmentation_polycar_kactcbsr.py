import numpy as np
#from pyDOE2 import lhs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

ds_fea1=pd.read_csv('d:/msk/lenin/polycarbonate/polycar_awjd.csv')#input file
data=ds_fea1
X=data.iloc[:,:4]
y1=data.iloc[:,4]
#y2=data.iloc[:,5]
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model1=LinearRegression().fit(X_poly, y1)
#model2=LinearRegression().fit(X_poly, y2)
y1_p=model1.predict(X_poly)
#y2_p=model2.predict(X_poly)
# R-squared
r_squared = r2_score(y1, y1_p)
print("R-squared:", r_squared)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y1, y1_p)
print("Mean Absolute Error:", mae)

# Mean Squared Error (MSE)
mse = mean_squared_error(y1, y1_p)
print("Mean Squared Error:", mse)
"""
# R-squared
r_squared_1 = r2_score(y2, y2_p)
print("R-squared1:", r_squared_1)

# Mean Absolute Error (MAE)
mae1 = mean_absolute_error(y2, y2_p)
print("Mean Absolute Error1:", mae1)

# Mean Squared Error (MSE)
mse1 = mean_squared_error(y2, y2_p)
print("Mean Squared Error1:", mse1)
"""
import random
def generate_random_values(num_values, lower_limits, upper_limits):
  random_data = []
  for _ in range(num_values):
    parameters = []
    for i in range(len(lower_limits)):
      # Generate a random value between the lower and upper limits (inclusive)
      parameters.append(random.uniform(lower_limits[i], upper_limits[i]))
    random_data.append(parameters)
  return random_data

# Example usage
num_values = 189
lower_limits = [250, 300, 200, 1.5]
upper_limits = [350, 500, 400, 2.5]
random_values = generate_random_values(num_values, lower_limits, upper_limits)
random_values_poly=poly.fit_transform(random_values)
y1r_p=model1.predict(random_values_poly)
#y2r_p=model2.predict(random_values_poly)
y1r_p=np.asarray(y1r_p)
y1r_p=pd.DataFrame(y1r_p)
#y2r_p=np.asarray(y2r_p)
#y2r_p=pd.DataFrame(y2r_p)
random_values=np.asarray(random_values)
random_values=pd.DataFrame(random_values)
aug_data=pd.concat([random_values, y1r_p], axis=1, join='inner')

aug_data.to_excel('d:/msk/lenin/polycarbonate/aug_ka2.xlsx')
print("coeff1:",model1.intercept_,model1.coef_)
#print("coeff2",model2.intercept_,model2.coef_)
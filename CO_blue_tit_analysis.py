'''
Method:
The research question is "To what extent is the growth of nestling blue tits (Cyanistes caeruleus) influenced by competition with siblings?", but a specific
hypothesis is not given. Therefore, I test the hypothesis: The growth of blue tits (Cyanistes caeruleus) is affected by nest co-habitation with siblings."
For paternity, I did not try try match chicks to sires because many of these data had missing values.
'''

# Import packages
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path

# Data import into pandas
data_folder = Path('./')
data_file = data_folder / 'blue_tit_data_updated_2020-04-18.csv'
df = pd.read_csv(data_file, sep=',', header=0, na_values='.')

# Examine the data
print(df.info(), '\n')

# Check for study size according to each year
print(df.hatch_year.value_counts(), '\n')

# How correlated are the two response variables
# Surprisingly, they are only moderately correlated, so I will test both models.
# I don't test for one accounting for the other, because I want to assess if the models
# explain size (day_14_tarsus_length) differently than mass (day_14_weight).
print(df[['day_14_weight','day_14_tarsus_length',]].corr())

# Check to see how weight and size vary across the three years
# They are significantly different, but differences aren't meaningful: the r2's are between 0.005-0.02
# Year is thereforenot included in the model
tarsus_year_lm = ols('day_14_tarsus_length ~ C(hatch_year)', data=df).fit()
print(sm.stats.anova_lm(tarsus_year_lm, typ=1))
print(tarsus_year_lm.rsquared)

weight_year_lm = ols('day_14_weight ~ C(hatch_year)', data=df).fit()
print(sm.stats.anova_lm(weight_year_lm, typ=1))
print(weight_year_lm.rsquared)

# Here I estimate the number of siblings per nest
# First impute missing data for rear_Cs_at_start_of_rearing
# Then, if home_or_away = home (1), then: rear_Cs_at_start_of_rearing - rear_Cs_in; else rear_Cs_in
df['rear_Cs_at_start_of_rearing'] = df['rear_Cs_at_start_of_rearing'].fillna(df['d14_rear_nest_brood_size'])
df['siblings'] = np.where(df['home_or_away']==1, df['rear_Cs_at_start_of_rearing'] - df['rear_Cs_in'], df['rear_Cs_in'])
df['non_siblings'] = df['rear_Cs_at_start_of_rearing'] - df['siblings']

# Reduce df to just the columns factors of interest that may affect growth-sibling interactions
# Extra-pair_paternity is not considered because there is a lot of missing data and I consider half-sibs as siblings.
# Drop rows with missing data
df_reduced = df[['day_14_weight', 'day_14_tarsus_length', 'siblings', 'non_siblings', 
    'chick_sex_molec', 'home_or_away']].dropna()

# Check that the data are roughly normal
# Nothing look too skewed...
# Not surprisingly, day_14_weight and day_14_tarsus_length look to be slightly skewed
df_reduced.hist()
plt.tight_layout()
plt.show()

# First, I consider size, using the proxy of 'day_14_tarsus_length'
# Create arrays for features and target variable
y_tarsus = df_reduced.day_14_tarsus_length.values
y_weight = df_reduced.day_14_weight.values
X = df_reduced.drop(['day_14_tarsus_length', 'day_14_weight'], axis=1).values

# Instantiate a lasso regressor for size (tarsus length): lasso
# Regularization (L1) will help:
#   avoid overfitting (lowers training accuracy, but improves test accuracy)
#   control for multicollinearity
#   helps with interpretability by minimizing irrelevant factors
# The loss function adds the absolute value of each coefficient times alpha (a constant parameter)
# The regressors are normalized before regression by subtracting the mean and dividing by the l2-norm
# First, use cross validation to find alpha value
# Fit the regressor to the data
lasso_reg_cv = LassoCV(cv=5, random_state=0).fit(X, y_tarsus)
print('Lasso regression CV score(r^2): ', lasso_reg_cv.score(X, y_tarsus))
print('Lasso regression CV alpha: ', lasso_reg_cv.alpha_)

# The model for size
lasso_reg = Lasso(alpha=lasso_reg_cv.alpha_, normalize=True)

# Fit the regressor to the data
lasso_reg.fit(X, y_tarsus)

# Compute and print the coefficients
lasso_reg_coef = lasso_reg.coef_
print('Lasso coefficients: ', lasso_reg_coef)
print('model r^2: ', lasso_reg.score(X, y_tarsus))

# Plot the coefficients
# According to the lasso algorithm (given the alpha value)
df_columns = df_reduced.drop(['day_14_tarsus_length', 'day_14_weight'], axis=1).columns.values
fig, ax = plt.subplots()
plt.bar(df_columns, lasso_reg_coef, color='black')
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
ax.set_title('Coefficients for tarsus length (Lasso Regression)')
plt.xticks(range(len(df_columns)), df_columns, rotation=60)

# Hide the right and top spines, and set margins
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.margins(0.02)
plt.show()

# Now, create a model for weight
lasso_reg_cv = LassoCV(cv=5, random_state=0).fit(X, y_weight)
print('Lasso regression CV score(r^2): ', lasso_reg_cv.score(X, y_weight))
print('Lasso regression CV alpha: ', lasso_reg_cv.alpha_)

# The model for weight
lasso_reg = Lasso(alpha=lasso_reg_cv.alpha_, normalize=True)

# Fit the regressor to the data
lasso_reg.fit(X, y_weight)

# Compute and print the coefficients
lasso_reg_coef = lasso_reg.coef_
print('Lasso coefficients: ', lasso_reg_coef)
print('model r^2: ', lasso_reg.score(X, y_weight))

# Plot the coefficients
# According to the lasso algorithm (given the alpha value)
df_columns = df_reduced.drop(['day_14_tarsus_length', 'day_14_weight'], axis=1).columns.values
fig, ax = plt.subplots()
plt.bar(df_columns, lasso_reg_coef, color='black')
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
ax.set_title('Coefficients for weight (Lasso Regression)')
plt.xticks(range(len(df_columns)), df_columns, rotation=60)

# Hide the right and top spines, and set margins
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.margins(0.02)
plt.show()

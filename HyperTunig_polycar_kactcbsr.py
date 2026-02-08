import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#2. load regressor libraries
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import uniform, randint
def relative_absolute_error(y_true, y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    numerator = np.abs(y_true - y_pred).sum()
    denominator = np.abs(y_true - np.mean(y_true)).sum()
    rae = numerator / denominator
    return rae

def relative_squared_error(y_true, y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - np.mean(y_true)) ** 2).sum()
    rse = numerator / denominator
    return rse

def correlation_coefficient(y_true, y_pred):
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    cc = np.corrcoef(y_true, y_pred)[0, 1]
    return cc


#3. loading data set
ds_fea1=pd.read_csv('d:/msk/lenin/polycarbonate/mldata_ka2.csv')#input file
ds_fea=ds_fea1.iloc[:,0:4];#for four parameters
ds_tar_end=ds_fea1.iloc[:,4]#4 for end 5 for exd and 6 for erd and 7 for taper

#cname=ds_fea1.columns
#sns.pairplot(ds_fea1, hue = cname[len(cname)-1])

#4. splitting test and training data
xtrain, xtest, ytrain, ytest = train_test_split(ds_fea, ds_tar_end, test_size = 0.25, random_state = 2)
#-----Ada Boost Regressor Hyper parameter tuning ----------------
param_dist_abr = {
    'n_estimators': randint(50, 200),  # Number of weak learners
    'learning_rate': uniform(0.01, 1.0)  # Contribution of each weak learner
}
param_dist_abr1 = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1],   
}
param_dist_xgb = {
    'max_depth': randint(3, 10),  # Maximum tree depth
    'learning_rate': uniform(0.01, 1.0),  # Learning rate
    'n_estimators': randint(50, 200),  # Number of trees
    'gamma': uniform(0, 0.5),  # Minimum loss reduction required to make a further partition on a leaf node
    'subsample': uniform(0.5, 0.5),  # Subsample ratio of the training instances
    'colsample_bytree': uniform(0.5, 0.5),  # Subsample ratio of columns when constructing each tree
}
param_dist_xgb1 = {
    'max_depth': [3,5,7,9],  # Maximum tree depth
    'learning_rate': [0.01, 0.1, 0.5, 1],  # Learning rate
    'n_estimators': [50, 100, 150, 200],  # Number of trees
    'gamma': np.random.uniform(0, 0.5, size=4).tolist(),  # Minimum loss reduction required to make a further partition on a leaf node
    'subsample': np.random.uniform(0.5, 0.5, size=4).tolist(),  # Subsample ratio of the training instances
    'colsample_bytree': np.random.uniform(0.5, 0.5, size=4).tolist(),  # Subsample ratio of columns when constructing each tree
}
param_dist_dt = {
    'max_depth': randint(1, 20),  # Maximum depth of the tree
    'min_samples_split': randint(2, 20),  # Minimum number of samples required to split an internal node
    'min_samples_leaf': randint(1, 20),  # Minimum number of samples required to be at a leaf node
}
param_dist_dt1 = {
    'max_depth': [1,5,10,20],  # Maximum depth of the tree
    'min_samples_split': [2,7,13, 20],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1,5,15,20],  # Minimum number of samples required to be at a leaf node
}
param_dist_rf= {
    'n_estimators': randint(10, 500),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    #'max_features': ['auto', 'sqrt', 'log2']
}
param_dist_rf1= {
    'n_estimators': [10, 200, 500],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    #'max_features': ['auto', 'sqrt', 'log2']
}
param_dist_svr = {
    'C': uniform(0.1, 10),  # Regularization parameter
    'epsilon': uniform(0.01, 0.1),  # Epsilon in the epsilon-SVR model
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'degree': randint(1, 10),  # Degree of the polynomial kernel function (only for poly kernel)
}
param_dist_svr1 = {
    'C': np.random.uniform(0.1, 10, size=4).tolist(),  # Regularization parameter
    'epsilon': np.random.uniform(0.01, 0.1, size=4).tolist(),  # Epsilon in the epsilon-SVR model
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'degree': [1, 4,7,10],  # Degree of the polynomial kernel function (only for poly kernel)
}
models = []
models.append(('ABR', AdaBoostRegressor()))
models.append(('XGB', XGBRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('RF', RandomForestRegressor()))
#models.append(('SVR', SVR()))
hy_par=[]
hy_par.append(('ABR',param_dist_abr))
hy_par.append(('XGB',param_dist_xgb))
hy_par.append(('DT',param_dist_dt))
hy_par.append(('RF',param_dist_rf))
#hy_par.append(('SVR',param_dist_svr))
hy_par1=[]
hy_par1.append(('ABR',param_dist_abr1))
hy_par1.append(('XGB',param_dist_xgb1))
hy_par1.append(('DT',param_dist_dt1))
hy_par1.append(('RF',param_dist_rf1))
#hy_par1.append(('SVR',param_dist_svr1))
hypar=[]
for hyp in hy_par:
    hypar.append(hyp[1])
hypar1=[]
for hyp1 in hy_par1:
    hypar1.append(hyp1[1])
i=0
or2_tr=[]
or2_tt=[]
or2=[]
ormse=[]
omse=[]
omae=[]
orse=[]
orae=[]
occ=[]

ormse_tr=[]
omse_tr=[]
omae_tr=[]
orse_tr=[]
orae_tr=[]
occ_tr=[]
ormse_tt=[]
omse_tt=[]
omae_tt=[]
orse_tt=[]
orae_tt=[]
occ_tt=[]

or2_tr_rs=[]
or2_tt_rs=[]
or2_rs=[]
ormse_rs=[]
omse_rs=[]
omae_rs=[]
orse_rs=[]
orae_rs=[]
occ_rs=[]
ormse_tr_rs=[]
omse_tr_rs=[]
omae_tr_rs=[]
orse_tr_rs=[]
orae_tr_rs=[]
occ_tr_rs=[]
ormse_tt_rs=[]
omse_tt_rs=[]
omae_tt_rs=[]
orse_tt_rs=[]
orae_tt_rs=[]
occ_tt_rs=[]
bst_para_rs=[]
bst_para=[]
for mdl in models:
    print(mdl[1])
    model1 = mdl[1]
    #param_dist=hy_par[i]
    #param_dist1=hy_par1[i]
    random_search = RandomizedSearchCV(model1, param_distributions=hypar[i], n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=42)
    # Perform RandomizedSearchCV
    random_search.fit(xtrain, ytrain)
    # Get the best estimator
    best_adaboost_reg = random_search.best_estimator_
    bst_para_rs.append(best_adaboost_reg)
    # Make predictions on the test set
    y_test = best_adaboost_reg.predict(xtest)
    y_train = best_adaboost_reg.predict(xtrain)
    y_all=best_adaboost_reg.predict(ds_fea)
    or2_rs.append(r2_score(ds_tar_end,y_all))
    or2_tr_rs.append(r2_score(ytrain,y_train))
    or2_tt_rs.append(r2_score(ytest,y_test))
    
    omae_tr_rs.append(mean_absolute_error(ytrain,y_train))
    omae_tt_rs.append(mean_absolute_error(ytest,y_test))
    omae_rs.append(mean_absolute_error(ds_tar_end,y_all))
    
    omse_tr_rs.append(mean_squared_error(ytrain,y_train))
    omse_tt_rs.append(mean_squared_error(ytest,y_test))
    omse_rs.append(mean_squared_error(ds_tar_end,y_all))
    
    ormse_tr_rs.append(np.sqrt(mean_squared_error(ytrain,y_train)))
    ormse_tt_rs.append(np.sqrt(mean_squared_error(ytest,y_test)))
    ormse_rs.append(np.sqrt(mean_squared_error(ds_tar_end,y_all)))
    
    occ_tr_rs.append(correlation_coefficient(ytrain,y_train))
    occ_tt_rs.append(correlation_coefficient(ytest,y_test))
    occ_rs.append(correlation_coefficient(ds_tar_end,y_all))
    
    orae_tr_rs.append(relative_absolute_error(ytrain,y_train))
    orae_tt_rs.append(relative_absolute_error(ytest,y_test))
    orae_rs.append(relative_absolute_error(ds_tar_end,y_all))
   
    orse_tr_rs.append(relative_squared_error(ytrain,y_train))
    orse_tt_rs.append(relative_squared_error(ytest,y_test))
    orse_rs.append(relative_squared_error(ds_tar_end,y_all))
    
    #print(mdl[0],"-RandomSCV Best Paraemters:",best_adaboost_reg, "R2-Training:",or2_tr_rs,"R2-Test:",or2_tt_rs)
    grid_search = GridSearchCV(estimator=model1, param_grid=hypar1[i], cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    # Fit the grid search to the data
    grid_search.fit(xtrain, ytrain)
    best_rf = grid_search.best_estimator_
    bst_para.append(best_rf)
    y_train1 = best_rf.predict(xtrain)
    y_test1 = best_rf.predict(xtest)
    y_all1=best_rf.predict(ds_fea)
    or2_tr.append(r2_score(ytrain,y_train1))
    or2_tt.append(r2_score(ytest,y_test1))
    or2.append(r2_score(ds_tar_end,y_all1))
    
    omae_tr.append(mean_absolute_error(ytrain,y_train1))
    omae_tt.append(mean_absolute_error(ytest,y_test1))
    omae.append(mean_absolute_error(ds_tar_end,y_all1))
    
    omse_tr.append(mean_squared_error(ytrain,y_train1))
    omse_tt.append(mean_squared_error(ytest,y_test1))
    omse.append(mean_squared_error(ds_tar_end,y_all1))
    
    ormse_tr.append(np.sqrt(mean_squared_error(ytrain,y_train1)))
    ormse_tt.append(np.sqrt(mean_squared_error(ytest,y_test1)))
    ormse.append(np.sqrt(mean_squared_error(ds_tar_end,y_all1)))
    
    occ_tr.append(correlation_coefficient(ytrain,y_train1))
    occ_tt.append(correlation_coefficient(ytest,y_test1))
    occ.append(correlation_coefficient(ds_tar_end,y_all1))
    
    orae_tr.append(relative_absolute_error(ytrain,y_train1))
    orae_tt.append(relative_absolute_error(ytest,y_test1))
    orae.append(relative_absolute_error(ds_tar_end,y_all1))
   
    orse_tr.append(relative_squared_error(ytrain,y_train1))
    orse_tt.append(relative_squared_error(ytest,y_test1))
    orse.append(relative_squared_error(ds_tar_end,y_all1))

    i=i+1
#------------------------------------- end -----
or2_rs=np.asarray(or2_rs)
or2_rs=pd.DataFrame(or2_rs)
or2_tr_rs=np.asarray(or2_tr_rs)
or2_tr_rs=pd.DataFrame(or2_tr_rs)
or2_tt_rs=np.asarray(or2_tt_rs)
or2_tt_rs=pd.DataFrame(or2_tt_rs)

omse_tr_rs=np.asarray(omse_tr_rs)
omse_tr_rs=pd.DataFrame(omse_tr_rs)
omse_tt_rs=np.asarray(omse_tt_rs)
omse_tt_rs=pd.DataFrame(omse_tt_rs)
omse_rs=np.asarray(omse_rs)
omse_rs=pd.DataFrame(omse_rs)

omae_tr_rs=np.asarray(omae_tr_rs)
omae_tr_rs=pd.DataFrame(omae_tr_rs)
omae_tt_rs=np.asarray(omae_tt_rs)
omae_tt_rs=pd.DataFrame(omae_tt_rs)
omae_rs=np.asarray(omae_rs)
omae_rs=pd.DataFrame(omae_rs)

ormse_tr_rs=np.asarray(ormse_tr_rs)
ormse_tr_rs=pd.DataFrame(ormse_tr_rs)
ormse_tt_rs=np.asarray(ormse_tt_rs)
ormse_tt_rs=pd.DataFrame(ormse_tt_rs)
ormse_rs=np.asarray(ormse_rs)
ormse_rs=pd.DataFrame(ormse_rs)

orse_tr_rs=np.asarray(orse_tr_rs)
orse_tr_rs=pd.DataFrame(orse_tr_rs)
orse_tt_rs=np.asarray(orse_tt_rs)
orse_tt_rs=pd.DataFrame(orse_tt_rs)
orse_rs=np.asarray(orse_rs)
orse_rs=pd.DataFrame(orse_rs)

orae_tr_rs=np.asarray(orae_tr_rs)
orae_tr_rs=pd.DataFrame(orae_tr_rs)
orae_tt_rs=np.asarray(orae_tt_rs)
orae_tt_rs=pd.DataFrame(orae_tt_rs)
orae_rs=np.asarray(orae_rs)
orae_rs=pd.DataFrame(orae_rs)

occ_tr_rs=np.asarray(occ_tr_rs)
occ_tr_rs=pd.DataFrame(occ_tr_rs)
occ_tt_rs=np.asarray(occ_tt_rs)
occ_tt_rs=pd.DataFrame(occ_tt_rs)
occ_rs=np.asarray(occ_rs)
occ_rs=pd.DataFrame(occ_rs)

rslt_rs=pd.concat([or2_rs, omse_rs, ormse_rs,omae_rs,orse_rs,orae_rs,occ_rs], axis=1, join='inner')
rslt_rs.to_excel('d:/msk/lenin/polycarbonate/polycar_ka2_alldata_rs_4ml_1.xlsx')

rslt_tr_rs=pd.concat([or2_tr_rs, omse_tr_rs, ormse_tr_rs,omae_tr_rs,orse_tr_rs,orae_tr_rs,occ_tr_rs], axis=1, join='inner')
rslt_tr_rs.to_excel('d:/msk/lenin/polycarbonate/polycar_ka2_trdata_rs_4ml_1.xlsx')

rslt_tt_rs=pd.concat([or2_tt_rs, omse_tt_rs, ormse_tt_rs,omae_tt_rs,orse_tt_rs,orae_tt_rs,occ_tt_rs], axis=1, join='inner')
rslt_tt_rs.to_excel('d:/msk/lenin/polycarbonate/polycar_ka2_ttdata_rs_4ml_1.xlsx')
#-------grid search
or2=np.asarray(or2)
or2=pd.DataFrame(or2)
or2_tr=np.asarray(or2_tr)
or2_tr=pd.DataFrame(or2_tr)
or2_tt=np.asarray(or2_tt)
or2_tt=pd.DataFrame(or2_tt)

omse_tr=np.asarray(omse_tr)
omse_tr=pd.DataFrame(omse_tr)
omse_tt=np.asarray(omse_tt)
omse_tt=pd.DataFrame(omse_tt)
omse=np.asarray(omse)
omse=pd.DataFrame(omse)

omae_tr=np.asarray(omae_tr)
omae_tr=pd.DataFrame(omae_tr)
omae_tt=np.asarray(omae_tt)
omae_tt=pd.DataFrame(omae_tt)
omae=np.asarray(omae)
omae=pd.DataFrame(omae)

ormse_tr=np.asarray(ormse_tr)
ormse_tr=pd.DataFrame(ormse_tr)
ormse_tt=np.asarray(ormse_tt)
ormse_tt=pd.DataFrame(ormse_tt)
ormse=np.asarray(ormse)
ormse=pd.DataFrame(ormse)

orse_tr=np.asarray(orse_tr)
orse_tr=pd.DataFrame(orse_tr)
orse_tt=np.asarray(orse_tt)
orse_tt=pd.DataFrame(orse_tt)
orse=np.asarray(orse)
orse=pd.DataFrame(orse)

orae_tr=np.asarray(orae_tr)
orae_tr=pd.DataFrame(orae_tr)
orae_tt=np.asarray(orae_tt)
orae_tt=pd.DataFrame(orae_tt)
orae=np.asarray(orae)
orae=pd.DataFrame(orae)

occ_tr=np.asarray(occ_tr)
occ_tr=pd.DataFrame(occ_tr)
occ_tt=np.asarray(occ_tt)
occ_tt=pd.DataFrame(occ_tt)
occ=np.asarray(occ)
occ=pd.DataFrame(occ)

rslt=pd.concat([or2, omse, ormse,omae,orse,orae,occ], axis=1, join='inner')
rslt.to_excel('d:/msk/lenin/polycarbonate/polycar_ka2_alldata_gs_4ml_1.xlsx')

rslt_tr=pd.concat([or2_tr, omse_tr, ormse_tr,omae_tr,orse_tr,orae_tr,occ_tr], axis=1, join='inner')
rslt_tr.to_excel('d:/msk/lenin/polycarbonate/polycar_ka2_trdata_gs_4ml_1.xlsx')

rslt_tt=pd.concat([or2_tt, omse_tt, ormse_tt,omae_tt,orse_tt,orae_tt,occ_tt], axis=1, join='inner')
rslt_tt.to_excel('d:/msk/lenin/polycarbonate/polycar_ka2_ttdata_gs_4ml_1.xlsx')

bst_para_rs=np.asarray(bst_para_rs)
bst_para_rs=pd.DataFrame(bst_para_rs)
bst_para_rs.to_excel('d:/msk/lenin/polycarbonate/polycar_ka2_bsthyparameters_rs_4ml_1.xlsx')
bst_para=np.asarray(bst_para)
bst_para=pd.DataFrame(bst_para)

bst_para.to_excel('d:/msk/lenin/polycarbonate/polycar_ka2_bsthyparameters_gs_4ml_1.xlsx')

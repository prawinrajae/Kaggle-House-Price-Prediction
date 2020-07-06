
#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sb
#Importing CSV files
dataset=pd.read_csv('/Users/prawinrajae/Downloads/House-Price-Prediction-master/train.csv')
dataset1=pd.read_csv('/Users/prawinrajae/Downloads/House-Price-Prediction-master/test.csv')
dataset.info()
#Shapes of datas
print('train data shape: ', dataset.shape)
print('test data shape: ', dataset1.shape)

#Null Values
total = dataset.isnull().sum().sort_values(ascending=False)
total1 = dataset1.isnull().sum().sort_values(ascending=False)

## plotting distribution of target feature
sb.distplot(dataset['SalePrice'])
plt.show()
#Target feature is not normally distributed and shows positive skewness. 

#  correlation matrix
sb.set(rc={'figure.figsize':(12,8)})
correlation_matrix = dataset.corr()

k = 10  #number of variables for heatmap
cols = correlation_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(dataset[cols].values.T)
sb.set(font_scale=1.25)
hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
'''From above heatmap we can see features which have high correlations with target feature but low correlation 
#among dependent features.'''


##Data Preprocessing
dataset=dataset.drop("PoolQC",axis=1)
dataset=dataset.drop("MiscFeature",axis=1)
dataset=dataset.drop("Fence",axis=1)
dataset=dataset.drop("FireplaceQu",axis=1)
dataset=dataset.drop("Alley",axis=1)
dataset.LotFrontage=dataset["LotFrontage"].fillna(dataset["LotFrontage"].mean())
dataset.MasVnrType=dataset["MasVnrType"].fillna(dataset["MasVnrType"].mode()[0])
dataset.MasVnrArea=dataset["MasVnrArea"].fillna(dataset["MasVnrArea"].mean())
dataset.BsmtQual =dataset["BsmtQual"].fillna(dataset["BsmtQual"].mode()[0])
dataset.BsmtCond =dataset["BsmtCond"].fillna(dataset["BsmtCond"].mode()[0])
dataset.BsmtCond =dataset["BsmtCond"].fillna(dataset["BsmtCond"].mode()[0])
dataset.BsmtExposure =dataset["BsmtExposure"].fillna(dataset["BsmtExposure"].mode()[0])
dataset.BsmtFinType2 =dataset["BsmtFinType2"].fillna(dataset["BsmtFinType2"].mode()[0])
dataset.BsmtFinType1 =dataset["BsmtFinType1"].fillna(dataset["BsmtFinType1"].mode()[0])
dataset.Electrical =dataset["Electrical"].fillna(dataset["Electrical"].mode()[0])
dataset.GarageType =dataset["GarageType"].fillna(dataset["GarageType"].mode()[0])
dataset.GarageYrBlt=dataset["GarageYrBlt"].fillna(dataset["GarageYrBlt"].mean())
dataset.GarageFinish =dataset["GarageFinish"].fillna(dataset["GarageFinish"].mode()[0])
dataset.GarageQual =dataset["GarageQual"].fillna(dataset["GarageQual"].mode()[0])
dataset.GarageCond =dataset["GarageCond"].fillna(dataset["GarageCond"].mode()[0])
####
dataset1.info()
dataset1=dataset1.drop("PoolQC",axis=1)
dataset1=dataset1.drop("MiscFeature",axis=1)
dataset1=dataset1.drop("Fence",axis=1)
dataset1=dataset1.drop("FireplaceQu",axis=1)
dataset1=dataset1.drop("Alley",axis=1)
dataset1.LotFrontage=dataset1["LotFrontage"].fillna(dataset1["LotFrontage"].mean())
dataset1.MasVnrType=dataset1["MasVnrType"].fillna(dataset1["MasVnrType"].mode()[0])
dataset1.MasVnrArea=dataset1["MasVnrArea"].fillna(dataset1["MasVnrArea"].mean())
dataset1.BsmtQual =dataset1["BsmtQual"].fillna(dataset1["BsmtQual"].mode()[0])
dataset1.BsmtCond =dataset1["BsmtCond"].fillna(dataset1["BsmtCond"].mode()[0])
dataset1.BsmtCond =dataset1["BsmtCond"].fillna(dataset1["BsmtCond"].mode()[0])
dataset1.BsmtExposure =dataset1["BsmtExposure"].fillna(dataset1["BsmtExposure"].mode()[0])
dataset1.BsmtFinType2 =dataset1["BsmtFinType2"].fillna(dataset1["BsmtFinType2"].mode()[0])
dataset1.BsmtFinType1 =dataset1["BsmtFinType1"].fillna(dataset1["BsmtFinType1"].mode()[0])
dataset1.Electrical =dataset1["Electrical"].fillna(dataset1["Electrical"].mode()[0])
dataset1.GarageType =dataset1["GarageType"].fillna(dataset1["GarageType"].mode()[0])
dataset1.GarageYrBlt=dataset1["GarageYrBlt"].fillna(dataset1["GarageYrBlt"].mean())
dataset1.GarageFinish =dataset1["GarageFinish"].fillna(dataset1["GarageFinish"].mode()[0])
dataset1.GarageQual =dataset1["GarageQual"].fillna(dataset1["GarageQual"].mode()[0])
dataset1.GarageCond =dataset1["GarageCond"].fillna(dataset1["GarageCond"].mode()[0])
dataset1.MSZoning =dataset1["MSZoning"].fillna(dataset1["MSZoning"].mode()[0])
dataset1.Utilities =dataset1["Utilities"].fillna(dataset1["Utilities"].mode()[0])
dataset1.Exterior1st =dataset1["Exterior1st"].fillna(dataset1["Exterior1st"].mode()[0])
dataset1.Exterior2nd =dataset1["Exterior2nd"].fillna(dataset1["Exterior2nd"].mode()[0])
dataset1.BsmtFinSF1=dataset1["BsmtFinSF1"].fillna(dataset1["BsmtFinSF1"].mean())
dataset1.BsmtFinSF2=dataset1["BsmtFinSF2"].fillna(dataset1["BsmtFinSF2"].mean())
dataset1.BsmtUnfSF=dataset1["BsmtUnfSF"].fillna(dataset1["BsmtUnfSF"].mean())
dataset1.TotalBsmtSF=dataset1["TotalBsmtSF"].fillna(dataset1["TotalBsmtSF"].mean())
dataset1.BsmtFullBath =dataset1["BsmtFullBath"].fillna(dataset1["BsmtFullBath"].mode()[0])
dataset1.BsmtHalfBath=dataset1["BsmtHalfBath"].fillna(dataset1["BsmtHalfBath"].mode()[0])
dataset1.KitchenQual=dataset1["KitchenQual"].fillna(dataset1["KitchenQual"].mode()[0])
dataset1.Functional=dataset1["Functional"].fillna(dataset1["Functional"].mode()[0])
dataset1.GarageCars=dataset1["GarageCars"].fillna(dataset1["GarageCars"].mode()[0])
dataset1.GarageArea=dataset1["GarageArea"].fillna(dataset1["GarageArea"].mean())
dataset1.SaleType=dataset1["SaleType"].fillna(dataset1["SaleType"].mode()[0])

########
#test&train
x_train= dataset.iloc[:,:-1]
y_train=dataset.iloc[:,-1]
x_test=dataset1

# Get dummies
#for train

dummies=pd.get_dummies(dataset.loc[:,['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']])
dummies.info()
new_dummy=dummies.drop(['MSZoning_C (all)','Street_Pave','LotShape_IR1','LandContour_Bnk','Utilities_AllPub','LotConfig_Corner','LandSlope_Gtl','Neighborhood_OldTown','Condition1_Artery','Condition2_Artery','BldgType_1Fam','HouseStyle_1.5Fin','RoofStyle_Flat','RoofMatl_CompShg','Exterior1st_MetalSd','Exterior2nd_MetalSd','MasVnrType_Stone','ExterQual_Ex','ExterCond_TA','Foundation_BrkTil','BsmtQual_Ex','BsmtCond_Fa','BsmtExposure_Av','BsmtFinType1_ALQ','BsmtFinType2_ALQ','Heating_GasA','HeatingQC_Ex','CentralAir_N','Electrical_FuseA','KitchenQual_Ex','Functional_Maj1','GarageType_Attchd','GarageFinish_Fin','GarageQual_Fa','GarageCond_Ex','PavedDrive_N','SaleType_COD','SaleCondition_Abnorml'],1)
x_Train=x_train.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'],1)
x_train=pd.concat([x_Train,new_dummy], axis=1)



#for test
dummies1=pd.get_dummies(dataset1.loc[:,['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']])
dummies1.info()
new1_dummy=dummies1.drop(['MSZoning_C (all)','Street_Pave','LotShape_IR1','LandContour_Bnk','Utilities_AllPub','LotConfig_Corner','LandSlope_Gtl','Neighborhood_OldTown','Condition1_Artery','Condition2_Artery','BldgType_1Fam','HouseStyle_1.5Fin','RoofStyle_Flat','RoofMatl_CompShg','Exterior1st_MetalSd','Exterior2nd_MetalSd','MasVnrType_Stone','ExterQual_Ex','ExterCond_TA','Foundation_BrkTil','BsmtQual_Ex','BsmtCond_Fa','BsmtExposure_Av','BsmtFinType1_ALQ','BsmtFinType2_ALQ','Heating_GasA','HeatingQC_Ex','CentralAir_N','Electrical_FuseA','KitchenQual_Ex','Functional_Maj1','GarageType_Attchd','GarageFinish_Fin','GarageQual_Fa','GarageCond_Ex','PavedDrive_N','SaleType_COD','SaleCondition_Abnorml'],1)
x_Test=x_test.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'],1)
x_test=pd.concat([x_Test,new1_dummy], axis=1)





##vif
def vif_calc(x):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    x_train['intercept']=1
    vif=pd.DataFrame()
    vif['variables']=x_train.columns
    vif['vif'] = [variance_inflation_factor(x_train.values,i) for i in range (0,x_train.shape[1])]
    return(vif)
vif=vif_calc(x_train)

x_train=x_train.drop(['BsmtFinSF1'],1)
x_train=x_train.drop(['1stFlrSF'],1)
x_train=x_train.drop(['RoofStyle_Gable'],1)
x_train=x_train.drop(['Exterior1st_CBlock'],1)    
x_train=x_train.drop(['GarageCond_TA'],1)      
x_train=x_train.drop(['Exterior2nd_VinylSd'],1)      
x_train=x_train.drop(['SaleType_New'],1) 
x_train=x_train.drop(['MSZoning_RL'],1) 
x_train=x_train.drop(['MSSubClass'],1)        
x_train=x_train.drop(['Exterior1st_CemntBd'],1)    
x_train=x_train.drop(['2ndFlrSF'],1)
x_train=x_train.drop(['ExterQual_TA'],1)
x_train=x_train.drop(['BsmtFinType2_Unf'],1)
x_train=x_train.drop(['YearBuilt'],1)
x_train=x_train.drop(['Exterior1st_HdBoard'],1)
x_train=x_train.drop(['GrLivArea'],1)
x_train=x_train.drop(['Condition2_Norm'],1)
x_train=x_train.drop(['BsmtQual_TA'],1)
x_train=x_train.drop(['Functional_Typ'],1)
x_train=x_train.drop(['Neighborhood_Somerst'],1)
x_train=x_train.drop(['KitchenQual_TA'],1)
x_train=x_train.drop(['TotalBsmtSF'],1)
x_train=x_train.drop(['Exterior2nd_Wd Sdng'],1)
x_train=x_train.drop(['Foundation_PConc'],1)
x_train=x_train.drop(['GarageCars'],1)
x_train=x_train.drop(['MasVnrType_None'],1)
x_train=x_train.drop(['HouseStyle_1Story'],1)
x_train=x_train.drop(['Neighborhood_NAmes'],1)
x_train=x_train.drop(['Exterior1st_AsbShng'],1)
x_train=x_train.drop(['SaleType_WD'],1)
x_train=x_train.drop(['Condition1_Norm'],1)
x_train=x_train.drop(['TotRmsAbvGrd'],1)
x_train=x_train.drop(['GarageYrBlt'],1)
x_train=x_train.drop(['OverallQual'],1)
x_train=x_train.drop(['BsmtFinType1_Unf'],1)
x_train=x_train.drop(['KitchenAbvGr'],1)
x_train=x_train.drop(['GarageFinish_Unf'],1)
x_train=x_train.drop(['Exterior2nd_Plywood'],1)
x_train=x_train.drop(['Electrical_Mix'],1)
x_train=x_train.drop(['Exterior2nd_Stucco'],1)
x_train=x_train.drop(['Exterior1st_VinylSd'],1)
x_train=x_train.drop(['Exterior2nd_Brk Cmn'],1)
x_train=x_train.drop(['BsmtCond_TA'],1)
x_train=x_train.drop(['RoofStyle_Shed'],1)
x_train=x_train.drop(['LandContour_Lvl'],1)
x_train=x_train.drop(['FullBath'],1)
x_train=x_train.drop(['Utilities_NoSeWa','Condition2_RRAn','Condition2_RRNn','HouseStyle_2.5Fin','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Exterior1st_ImStucc','Exterior1st_Stone','Exterior2nd_Other','Heating_Floor',
                      'Heating_OthW','GarageQual_Ex','Condition2_RRAe'],1)
    
x_train=x_train.drop(['intercept'],1)

x_test=x_test.drop(['BsmtFinSF1','1stFlrSF','RoofStyle_Gable','Exterior1st_CBlock','GarageCond_TA','Exterior2nd_VinylSd','SaleType_New','MSZoning_RL','MSSubClass','Exterior1st_CemntBd','2ndFlrSF','ExterQual_TA','BsmtFinType2_Unf','YearBuilt',
                    'Exterior1st_HdBoard','GrLivArea','Condition2_Norm','BsmtQual_TA','Functional_Typ','Neighborhood_Somerst','KitchenQual_TA','TotalBsmtSF','Exterior2nd_Wd Sdng','Foundation_PConc','GarageCars','MasVnrType_None','HouseStyle_1Story',
                    'Neighborhood_NAmes','Exterior1st_AsbShng','SaleType_WD','Condition1_Norm','TotRmsAbvGrd','GarageYrBlt','OverallQual','BsmtFinType1_Unf','KitchenAbvGr','GarageFinish_Unf','Exterior2nd_Plywood','Exterior2nd_Stucco',
                    'Exterior1st_VinylSd','Exterior2nd_Brk Cmn','BsmtCond_TA','RoofStyle_Shed','LandContour_Lvl','FullBath'],1)




#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regression=LinearRegression()

regression.fit(x_train,y_train)
regression.coef_
regression.intercept_
####y_pred
y_pred=regression.predict(x_test)

# check score
print('Train Score: ', regression.score(x_train,y_train))
print('Test Score: ', regression.score(x_test,y_pred))




##Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(x_train,y_train)
y_pred_ran = rf_reg.predict(x_test)


# check score
print('Train Score: ', rf_reg.score(x_train,y_train))
print('Test Score: ', rf_reg.score(x_test,y_pred))


 
####Regularization
    ####Lasso&Ridge
    
###Cross Validation
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
from sklearn.model_selection import cross_val_score
mse=cross_accuracy=cross_val_score(lin,x_train,y_train,cv=5,scoring='r2')

mse = np.mean(mse)
print(mse)

#ridge&lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()

parameter={'alpha':[0,1e-10,1e-15,1,2,3,10,50,100]}

ridge_grid=GridSearchCV(ridge,
                         param_grid=parameter,
                         scoring='r2',
                         cv=5)

ridge_grid.fit(x_train,y_train)
y_pred_ridge=ridge_grid.predict(x_test)

ridge_grid.best_score_
ridge_grid.best_params_
##
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso = Lasso()

parameter={'alpha':[0,1e-10,1e-15,1,2,3,10,50,100]}

lasso_grid=GridSearchCV(lasso,
                         param_grid=parameter,
                         scoring='r2',
                         cv=5)

lasso_grid.fit(x_train,y_train)

y_pred_lasso=lasso_grid.predict(x_test)
lasso_grid.best_score_
lasso_grid.best_params_







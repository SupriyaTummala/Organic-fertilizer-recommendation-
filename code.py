import pandas as pd
df=pd.read_csv("Train.csv")
print(df)
import matplotlib.pyplot as plt
plt.scatter(df['CultLand'],df['CropCultLand'])
plt.show()
cultland_med=df['CultLand'].median()
df['CultLand'] = df['CultLand'].astype(int)
new_df = df.copy()  # Create a copy of the original DataFrame
new_df['CultLand'] = new_df['CultLand'].replace(to_replace=800, value=20)
new_df['CultLand'][:2555]
new_df.iloc[:,11]
seed_med=crop['SeedlingsPerPit'].median()
print(seed_med)
crop['SeedlingsPerPit'] = crop['SeedlingsPerPit'].fillna(crop['SeedlingsPerPit'].median())
crop['SeedlingsPerPit']
crop['SeedlingsPerPit'][:132]
crop['SeedlingsPerPit'].median()
df=crop.copy()
df['SeedlingsPerPit'] = df['SeedlingsPerPit'].replace(to_replace=442, value=2)
df['SeedlingsPerPit'][:136]
df['TransplantingIrrigationHours'].median()
df_new=df.copy()
df_new['TransplantingIrrigationHours']=df_new['TransplantingIrrigationHours'].replace(to_replace=2000,value=df['TransplantingIrrigationHours'].median())
df_new['TransplantingIrrigationHours'][:2407]
df_new['TransplantingIrrigationHours']=df_new['TransplantingIrrigationHours'].fillna(df['TransplantingIrrigationHours'].median())
df_new['TransplantingIrrigationHours'][:2408]
df['TransIrriCost'].isnull()
df=df_new.copy()
df['TransIrriCost']=df['TransIrriCost'].replace(to_replace=6000,value=df_new['TransIrriCost'].median())
df['TransIrriCost']=df['TransIrriCost'].fillna(df_new['TransIrriCost'].median())
df['TransIrriCost'][:1000]
df1=df.copy()
df1['StandingWater']=df1['StandingWater'].replace(to_replace=[13,15],value=df['StandingWater'].median())
df1['StandingWater']=df1['StandingWater'].fillna(df['StandingWater'].median())
df1['StandingWater'][:2385]
df=df1.copy()
df['Ganaura']=df['Ganaura'].replace(to_replace=[500,600,800,900,1200,1400],value=df1['Ganaura'].median())
df['Ganaura']=df['Ganaura'].fillna(df1['Ganaura'].median())
df['Ganaura'][:1617]
df1=df.copy()
df1['CropOrgFYM']=df1['CropOrgFYM'].replace(to_replace=[300,600,500,1100,1200,1600,1500,2000,2200,1800,3000,2300,2500,4000],value=df['CropOrgFYM'].median())
df1['CropOrgFYM']=df1['CropOrgFYM'].fillna(df['CropOrgFYM'].median())
df=df1.copy()
df['BasalDAP']=df['BasalDAP'].replace(to_replace=100,value=df1['BasalDAP'].median())
df['BasalDAP']=df['BasalDAP'].fillna(df1['BasalDAP'].median())
df['BasalDAP'][:1579]
df1=df.copy()
df1['BasalUrea']=df1['BasalUrea'].replace(to_replace=[70,90],value=df['BasalUrea'].median())
df1['BasalUrea']=df1['BasalUrea'].fillna(df['BasalUrea'].median())
df=df1.copy()
df['1tdUrea']=df['1tdUrea'].replace(to_replace=[60,70,80,90],value=df1['1tdUrea'].median())
df['1tdUrea']=df['1tdUrea'].fillna(df1['1tdUrea'].median())
df1=df.copy()
df1['1appDaysUrea']=df1['1appDaysUrea'].replace(332,df['1appDaysUrea'].median())
df1['1appDaysUrea']=df1['1appDaysUrea'].fillna(df['1appDaysUrea'].median())
df=df1.copy()
df['2tdUrea']=df['2tdUrea'].replace([60,67],df1['2tdUrea'].median())
df['2tdUrea']=df['2tdUrea'].fillna(df1['2tdUrea'].median())
df1=df.copy()
df1['2appDaysUrea']=df1['2appDaysUrea'].replace(97,df['2appDaysUrea'].median())
df1['2appDaysUrea']=df1['2appDaysUrea'].fillna(df['2appDaysUrea'].median())
df=df1.copy()
df['Harv_hand_rent']=df['Harv_hand_rent'].replace(60000,df1['Harv_hand_rent'].median())
df['Harv_hand_rent']=df['Harv_hand_rent'].fillna(df1['Harv_hand_rent'].median())
df1=df.copy()
df1['Acre']=df1['Acre'].replace([2.1875,1.818,1.85185,1.925,1.6363,1.5909],df['Acre'].median())
df1['Acre']=df1['Acre'].fillna(df['Acre'].median())
df=df1.copy()
df['Yield']=df['Yield'].replace([16800,14400],df1['Yield'].median())
df['Yield']=df['Yield'].fillna(df1['Yield'].median())
df.isnull().sum()
mode_value = df['RcNursEstDate'].mode().values[0]
df['RcNursEstDate'].fillna(mode_value, inplace=True)
df['RcNursEstDate'][:9]
mode_value = df['TransDetFactor'].mode().values[0]
df['TransDetFactor'].fillna(mode_value, inplace=True)
df['TransDetFactor'][:9]
mode1 = df['NursDetFactor'].mode().values[0]
df['NursDetFactor'].fillna(mode1, inplace=True)
df['NursDetFactor'][:9]
mode1 = df['TransplantingIrrigationSource'].mode().values[0]
df['TransplantingIrrigationSource'].fillna(mode1, inplace=True)
df['TransplantingIrrigationSource'][:9]
mode1 = df['TransplantingIrrigationPowerSource'].mode().values[0]
df['TransplantingIrrigationPowerSource'].fillna(mode1, inplace=True)
df['TransplantingIrrigationPowerSource'][:9]
mode1 = df['OrgFertilizers'].mode().values[0]
df['OrgFertilizers'].fillna(mode1, inplace=True)
df['OrgFertilizers'][:9]
mode1 = df['PCropSolidOrgFertAppMethod'].mode().values[0]
df['PCropSolidOrgFertAppMethod'].fillna(mode1, inplace=True)
df['PCropSolidOrgFertAppMethod'][:9]
mode1 = df['CropbasalFerts'].mode().values[0]
df['CropbasalFerts'].fillna(mode1, inplace=True)
df['CropbasalFerts'][:15]
mode1 = df['FirstTopDressFert'].mode().values[0]
df['FirstTopDressFert'].fillna(mode1, inplace=True)
df['FirstTopDressFert'][10:30]
mode1 = df['MineralFertAppMethod'].mode().values[0]
df['MineralFertAppMethod'].fillna(mode1, inplace=True)
df['MineralFertAppMethod'][10:30]
mode1 = df['MineralFertAppMethod.1'].mode().values[0]
df['MineralFertAppMethod.1'].fillna(mode1, inplace=True)
df['MineralFertAppMethod.1'][10:30]
dftest = pd.read_csv("Test.csv")
column_to_replace = 'RcNursEstDate'
mode_value = dftest[column_to_replace].mode()[0]
dftest[column_to_replace].fillna(mode_value,inplace=True)
column_to_replace = 'NursDetFactor'
mode_value = dftest[column_to_replace].mode()[0]
dftest[column_to_replace].fillna(mode_value,inplace=True)
column_to_replace = 'TransDetFactor'
mode_value = dftest[column_to_replace].mode()[0]
dftest[column_to_replace].fillna(mode_value,inplace=True)
column_to_replace = 'TransplantingIrrigationSource'
mode_value = dftest[column_to_replace].mode()[0]
dftest[column_to_replace].fillna(mode_value,inplace=True)
column_to_replace = 'TransplantingIrrigationPowerSource'
mode_value = dftest[column_to_replace].mode()[0]
dftest[column_to_replace].fillna(mode_value,inplace=True)
column_to_replace = 'OrgFertilizers'
mode_value = dftest[column_to_replace].mode()[0]
dftest[column_to_replace].fillna(mode_value,inplace=True)
column_to_replace = 'PCropSolidOrgFertAppMethod'
mode_value = dftest[column_to_replace].mode()[0]
dftest[column_to_replace].fillna(mode_value,inplace=True)
column_to_replace = 'CropbasalFerts'
mode_value = dftest[column_to_replace].mode()[0]
dftest[column_to_replace].fillna(mode_value,inplace=True)
column_to_replace = 'FirstTopDressFert'
mode_value = dftest[column_to_replace].mode()[0]
column_to_replace = 'MineralFertAppMethod'
mode_value = dftest[column_to_replace].mode()[0]
dftest[column_to_replace].fillna(mode_value,inplace=True)
column_to_replace = 'SeedlingsPerPit'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = 'TransplantingIrrigationHours'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = 'TransIrriCost'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = 'StandingWater'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = 'Ganaura'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = 'CropOrgFYM'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = 'BasalDAP'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = 'BasalUrea'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = '1tdUrea'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = '1appDaysUrea'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = '2appDaysUrea'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
column_to_replace = 'Harv_hand_rent'
median_value = dftest[column_to_replace].median()
dftest[column_to_replace].fillna(median_value,inplace=True)
print(dftest)
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df['ID']=label_encoder.fit_transform(df['ID'])
#df['PCropSolidOrgFertAppMethod']=label_encoder.fit_transform(df['PCropSolidOrgFertAppMethod'])
#df['MineralFertAppMethod.1']=label_encoder.fit_transform(df['MineralFertAppMethod.1'])
#df['MineralFertAppMethod']=label_encoder.fit_transform(df['MineralFertAppMethod'])
df['District']=label_encoder.fit_transform(df['District'])
df['Block']=label_encoder.fit_transform(df['Block'])
df['LandPreparationMethod']=label_encoder.fit_transform(df['LandPreparationMethod'])
df['CropTillageDepth']=label_encoder.fit_transform(df['CropTillageDepth'])
df['CropTillageDate']=label_encoder.fit_transform(df['CropTillageDate'])
df['CropEstMethod']=label_encoder.fit_transform(df['CropEstMethod'])
df['RcNursEstDate']=label_encoder.fit_transform(df['RcNursEstDate'])
df['SeedingSowingTransplanting']=label_encoder.fit_transform(df['SeedingSowingTransplanting'])
df['NursDetFactor']=label_encoder.fit_transform(df['NursDetFactor'])
df['TransDetFactor']=label_encoder.fit_transform(df['TransDetFactor'])
df['TransplantingIrrigationSource']=label_encoder.fit_transform(df['TransplantingIrrigationSource'])
df['TransplantingIrrigationPowerSource']=label_encoder.fit_transform(df['TransplantingIrrigationPowerSource'])
df['OrgFertilizers']=label_encoder.fit_transform(df['OrgFertilizers'])
df['CropbasalFerts']=label_encoder.fit_transform(df['CropbasalFerts'])
df['FirstTopDressFert']=label_encoder.fit_transform(df['FirstTopDressFert'])
df['Harv_method']=label_encoder.fit_transform(df['Harv_method'])
df['Harv_date']=label_encoder.fit_transform(df['Harv_date'])
df['Threshing_method']=label_encoder.fit_transform(df['Threshing_method'])
df['Threshing_date']=label_encoder.fit_transform(df['Threshing_date'])
df['Stubble_use']=label_encoder.fit_transform(df['Stubble_use'])
df['ID']
df['PCropSolidOrgFertAppMethod']
dftest['ID']=label_encoder.fit_transform(dftest['ID'])
dftest['ID']
dftest['PCropSolidOrgFertAppMethod']=label_encoder.fit_transform(dftest['PCropSolidOrgFertAppMethod'])
dftest['MineralFertAppMethod']=label_encoder.fit_transform(dftest['MineralFertAppMethod'])
dftest['MineralFertAppMethod']=label_encoder.fit_transform(dftest['MineralFertAppMethod'])
dftest['District']=label_encoder.fit_transform(dftest['District'])
dftest['Block']=label_encoder.fit_transform(dftest['Block'])
dftest['LandPreparationMethod']=label_encoder.fit_transform(dftest['LandPreparationMethod'])
dftest['CropTillageDepth']=label_encoder.fit_transform(dftest['CropTillageDepth'])
dftest['CropTillageDate']=label_encoder.fit_transform(dftest['CropTillageDate'])
dftest['CropEstMethod']=label_encoder.fit_transform(dftest['CropEstMethod'])
dftest['RcNursEstDate']=label_encoder.fit_transform(dftest['RcNursEstDate'])
dftest['SeedingSowingTransplanting']=label_encoder.fit_transform(dftest['SeedingSowingTransplanting'])
dftest['NursDetFactor']=label_encoder.fit_transform(dftest['NursDetFactor'])
dftest['TransDetFactor']=label_encoder.fit_transform(dftest['TransDetFactor'])
dftest['TransplantingIrrigationSource']=label_encoder.fit_transform(dftest['TransplantingIrrigationSource'])
dftest['TransplantingIrrigationPowerSource']=label_encoder.fit_transform(dftest['TransplantingIrrigationPowerSource'])
dftest['OrgFertilizers']=label_encoder.fit_transform(dftest['OrgFertilizers'])
dftest['CropbasalFerts']=label_encoder.fit_transform(dftest['CropbasalFerts'])
dftest['FirstTopDressFert']=label_encoder.fit_transform(dftest['FirstTopDressFert'])
dftest['Harv_method']=label_encoder.fit_transform(dftest['Harv_method'])
dftest['Harv_date']=label_encoder.fit_transform(dftest['Harv_date'])
dftest['Threshing_method']=label_encoder.fit_transform(dftest['Threshing_method'])
dftest['Threshing_date']=label_encoder.fit_transform(dftest['Threshing_date'])
dftest['Stubble_use']=label_encoder.fit_transform(dftest['Stubble_use'])
df['RcNursEstDate']
df.isnull().sum()
df.dtypes
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
X = df.drop('Yield', axis=1) 
y = df['Yield']  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_components = 5  
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
scorer = make_scorer(lambda y_true, y_pred: -mean_squared_error(y_true, y_pred))
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = cross_val_score(rf_regressor, X_pca, y, cv=k_fold, scoring=scorer)
rmse_scores = [np.sqrt(-score) for score in mse_scores]
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean RMSE:", np.mean(rmse_scores))
print("Std RMSE:", np.std(rmse_scores))
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Random Forest Regressor with PCA')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.grid(True)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.show()

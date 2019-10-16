import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor



def manipulateData(data):
	
	#Drop Irrelavant Data
	data = data.drop(columns = ['Hair Color'])

	#Gender Manipulation
	data['Gender'] = data['Gender'].fillna('unknown')
	data['Gender'] = data['Gender'].replace('0','unknown')

	#Fill Null values of "Year of Record & Age" with mean of entire Dataset
	data['Year of Record'] = data['Year of Record'].fillna(round(data['Year of Record'].median()))
	data['Age'] = data['Age'].fillna(round(data['Age'].median()))

	# # Fill Null Value for "Profession"
	data['Profession'] = data['Profession'].fillna('unknown')

	#Fill Null values & 0 for Degree
	data['University Degree'] = data['University Degree'].replace('0','No')
	data['University Degree'] = data['University Degree'].fillna('Bachelor')
	

	#Encoding Categorical Values
	data['Profession'] = data['Profession'].str.upper() # Convert all values to lowercase
	data['Profession'] = data['Profession'].str.strip() # Remove White Space
	data['Profession'] = data['Profession'].str[:4]
	data['University Degree'] = data['University Degree'].str.lower() # Convert all values to uppercase
	data['University Degree'] = data['University Degree'].str.strip() # Remove White Space
	data['Country'] = data['Country'].str.strip() # Remove White Space
	
	
	##################Label Encoder with Result 146k#####################
	# label_encoder = preprocessing.LabelEncoder() 
	# data['Country']= label_encoder.fit_transform(data['Country']) 
	# data['Gender']= label_encoder.fit_transform(data['Gender']) 
	# data['Profession']= label_encoder.fit_transform(data['Profession']) 
	# data['University Degree']= label_encoder.fit_transform(data['University Degree'])


	# #################Encoding Style with Result 80k#####################
	# dummy_Country = pd.get_dummies(data["Country"]) # Country Encoded
	# dummy_Gender = pd.get_dummies(data["Gender"])  # Gender Encoded
	# dummy_Profession = pd.get_dummies(data["Profession"]) #Profession Encoded.
	# dummy_Degree = pd.get_dummies(data["University Degree"]) # Degree Encoded
	# data['Body Height [cm]'] = (data['Body Height [cm]'] - data['Body Height [cm]'].mean())/(data['Body Height [cm]'].std())
	# data['Year of Record'] = (data['Year of Record'] - data['Year of Record'].min())/(data['Year of Record'].max() - data['Year of Record'].min())
	data['Gender'] = data['Gender'].astype('category').cat.codes

	# #################    Target Mean Encoding 62K    #####################
	data['Country'] = data['Country'].map(data.groupby('Country')['Income in EUR'].mean())
	data['Profession'] = data['Profession'].map(data.groupby('Profession')['Income in EUR'].mean())
	data['University Degree'] = data['University Degree'].map(data.groupby('University Degree')['Income in EUR'].mean())
	data['Country'] = data['Country'].fillna(round(data['Country'].mean()))
	data['Profession'] = data['Profession'].fillna(round(data['Profession'].mean()))
 
	# data = pd.concat([data,dummy_Gender,dummy_Profession,dummy_Country,dummy_Degree],axis = 1)  #Concat all Dummy Data with Original Data
	# data = data.drop(columns = ['Country','Gender','Profession','University Degree']) #Drop all the non-Encoded Categorical Values
	
	return data

#import Training Dataset
data = pd.read_csv('~/MachineLearning/kaggle/Project1/tcd ml 2019-20 income prediction training (with labels).csv')
data.head(20)
#Income in EUR manipulation for trainning data
#data = data.drop(data[data['Income in EUR'] < 0].index)

#data = data.drop(data['Income in EUR'].idxmax())

random_data_std = data['Income in EUR'].std()
random_data_mean = data['Income in EUR'].mean()
anomaly_cut_off = random_data_std * 3

lower_limit  = random_data_mean - anomaly_cut_off 
upper_limit = random_data_mean + anomaly_cut_off
print(lower_limit)
print(upper_limit)

# Generate outliers
for outlier in data['Income in EUR']:
    if outlier > upper_limit or outlier < lower_limit:
    	mean = data['Income in EUR'].mean()
    	median = data['Income in EUR'].median()
    	print(mean)
    	data['Income in EUR'] = np.where(data['Income in EUR'] >upper_limit, mean, data['Income in EUR'])
    	data['Income in EUR'] = np.where(data['Income in EUR'] <lower_limit ,median, data['Income in EUR'])
print(data['Income in EUR'].mean())
data = data.drop(data[data['Income in EUR'] < 0].index)
print(data['Income in EUR'].mean())
#import Test Dataset

dataset = pd.read_csv('~/MachineLearning/kaggle/Project1/tcd ml 2019-20 income prediction test (without labels).csv')
data.head(20)

Merged_Dataset = pd.concat([data,dataset]) #Merge both the datasets
 
Merged_Dataset = manipulateData(Merged_Dataset) # Configure Dataset for Regression Model
prodData, trainData = [x for _, x in Merged_Dataset.groupby(Merged_Dataset['Income in EUR'] > 0)] # Split Dataset to Prod & Train
x = trainData.loc[:,trainData.columns != 'Income in EUR']
y = trainData['Income in EUR']

prod_Test = prodData.loc[:,prodData.columns != 'Income in EUR']


# ---------------------------Linear Regression Model ------------------------------------ #
# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 10)
# linearRegressor = LinearRegression()
# linearRegressor.fit(xTrain, yTrain)
# yPred =  linearRegressor.predict(xTest)
# print("Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(yTest, yPred)))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(yTest, yPred))

# yProdPrediction = linearRegressor.predict(prod_Test)

# #Configure Dataset and create a csv
# prodData['Income in EUR'] = yProdPrediction
# incomePrediction = prodData[['Instance','Income in EUR']]
# incomePrediction.to_csv('~/MachineLearning/kaggle/Income_Prediction.csv')

# ---------------------------Random Forest Model ------------------------------------ #
train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size = 0.1, random_state = 10)
rf = RandomForestRegressor(n_estimators = 100, random_state = 10)
rf.fit(train_features, train_labels);
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2))
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print("Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(test_labels, predictions)))

yProdPrediction = rf.predict(prod_Test)

# #Configure Dataset and create a csv
prodData['Income'] = yProdPrediction
incomePrediction = prodData[['Instance','Income']]
incomePrediction.to_csv('~/MachineLearning/kaggle/Income_PredictionNew2.csv',index = False)

# ---------------------------SVM ------------------------------------ #

# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 10)
# xgb_classifier = XGBClassifier()
# xgb_classifier.fit(xTrain,yTrain)
# yPred = xgb_classifier.predict(xTest)
# print("Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(yTest, yPred)))



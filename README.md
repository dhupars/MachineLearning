# MachineLearning

Machine Learning Model to Predict Income

## Required dependencies:

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install numpy, sklearn, pandas
```

## Steps to Predict Output
In file income.py, uncomment the model that you want to use for prediction (By default, it runs random forest)


Run the command python income.py

After a wait of few minutes the predicted result in generated in the file Income_PredictionNew2.csv

## Project Flow

1. Reads the dataset provided to train the model tcd ml 2019-20 income prediction training (with labels).csv
2. Reads the dataset on which income output is to be predicted tcd ml 2019-20 income prediction test (without labels).csv
3. Merge both the set set and pre-process data (remove outlier/target mean encoding)
4. Split the dataset into train and test dataset
5. Performs training of the model on the training dataset
6. Returns CSV for the prediction of income on the test dataset : Income_PredictionNew2.csv


## COVID-19 model

From the paper:
[Who should we test for COVID-19? A triage model built from national symptom surveys](https://www.medrxiv.org/content/10.1101/2020.05.18.20105569v2)
medRxiv 2020.05.18.20105569; doi: https://doi.org/10.1101/2020.05.18.20105569

The model predicts the probability of an individual to test positive in a COVID-19 PCR test, 
based on the following features:
* Age
* Gender
* Any of the prior medical conditions: Diabetes mellitus, Hypertension, Cardiovascular disease, Chronic lung disease, Chronic kidney disease, Malignancy (cancer) or Immunodeficiency
* General feeling
* Sore throat
* Cough
* Shortness of breath
* Smell or taste loss
* Fever (over 38 degrees celcius)

Python 3.7.6
sklearn version: 0.21.3
xgboost version: 1.0.2

Creating the models usage:
1. Update the response.json file with the answer to the features
(True means having the symptoms, at least one of the prior medical conditions, feeling well and being male, False means the opposite)
2. In the python file choose the model_path to be xgboost_model.sav or logistic_model.sav
3. Run the python file to get the prediction
- Should take seconds on an average PC

Running the models usage:
1. Choose which model you want to run and adjust the 'model' variable in the python file accordingly
2. Replace the simulated_data.csv file with real data for model creation
3. Run the python file to create the models
- Should take seconds on an average PC

Data:
'aggregated_data.csv' contains mean value for the symptoms above in the online version of the survey, aggregated by age group and gender
'predictions_model-name_model-type.csv' contains the true label, predicted probability by our model and predicted probability by the baseline model for each of the models

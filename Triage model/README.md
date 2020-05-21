## COVID-19 model

Model from the paper:
["Who should we test for COVID-19? A triage model built from national symptom surveys"](https://www.medrxiv.org/content/10.1101/2020.05.18.20105569v1) - Saar Shoer, Tal Karady, Ayya Keshet, Smadar Shilo, Hagai Rossman, Amir Gavrieli, Tomer Meir, Amit Lavon, Dmitry Kolobkov, Iris Kalka, Anastasia Godneva, Ori Cohen, Adam Kariv, Ori Hoch, Mushon Zer-Aviv, Noam Castel, Anat Ekka Zohar, Angela Irony, Benjamin Geiger, Dorit Hizi, Varda Shalev, Ran Balicer, Eran Segal

doi: https://doi.org/10.1101/2020.05.18.20105569

The model predicts the probability of individual to test positive in a COVID-19 PCR test, 
based on the following features:
* Age
* Gender
* Any of the prior medical conditions: Diabetes mellitus, Hypertension, Cardiovascular disease, Chronic lung disease, Chronic kidney disease, Malignancy (cancer) or Immunodeficiency
* General feeling
* Sore throat
* Cough
* Shortness of breath
* Smell or taste loss
* Fever (over 38 degrees celicius)

Usage:
1. Update the response.json file with the answer to the features
(True means having the symptoms, at least one of the prior medical conditions, feeling well and being male, False means the opposite)
2. In the python file choose the model_path to be xgboost_model.sav or logistic_model.sav
3. Run the python file to get the prediction

## aggregated data
'aggregated_data.csv' contains mean value for the symptoms above in the online version of the survey, aggregated by age group and gender.

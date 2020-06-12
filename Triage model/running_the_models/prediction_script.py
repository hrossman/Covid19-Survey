import pickle
import numpy as np
import pandas as pd

json_path = "response.json"
model_path = "logistic_model.sav"

AGE_GROUP_CUTOFFS = [0, 17, 30, 40, 50, 60, 70, 120]
AGE_GROUPS_TRANSFORMER = {1: 10, 2: 25, 3: 35, 4: 45, 5: 55, 6: 65, 7: 75}
AGE_COL = 'age_group'

X_COLS = ["gender", AGE_COL, "condition_any", "symptom_well", "symptom_sore_throat", "symptom_cough",
          "symptom_shortness_of_breath", "symptom_smell_or_taste_loss", "symptom_fever"]


def add_age_group(df):
    df[AGE_COL] = pd.cut(df['age'], bins=AGE_GROUP_CUTOFFS, labels=AGE_GROUPS_TRANSFORMER.values(), include_lowest=True, right=True)
    df[AGE_COL] = df[AGE_COL].astype(int)
    return df


def get_prediction(json_path, model_path):
    response_df = pd.read_json(json_path, lines=True)
    response_df = add_age_group(response_df)
    response_df = response_df[X_COLS].sort_index(axis=1)

    model = pickle.load(open(model_path, "rb"))
    predictions = model.predict_proba(response_df)
    predicted_probability = np.round(predictions[:, 1][0], 3)
    return predicted_probability


if __name__ == '__main__':
    print("The response probability to test positive according to our model is:", get_prediction(json_path, model_path))

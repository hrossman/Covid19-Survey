import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV

model = XGBClassifier  # alternative LogisticRegression


def create_model_from_dataset(model, X, y, cv=4, **kwargs):

    clf = model(**kwargs)
    y_pred = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    clf.fit(X, y)

    return clf, y_pred


def create_model_without_validation(model, dataset, x_cols, baseline_x_cols, random_state=10, **kwargs):

    X = dataset[x_cols].sort_index(axis=1)
    y = dataset['label'].values.ravel()

    fit_model, y_pred_vali = create_model_from_dataset(model, X, y, random_state=random_state, **kwargs)

    # base model
    X_base = dataset[baseline_x_cols].sort_index(axis=1)

    fit_model_base, y_pred_vali_base = create_model_from_dataset(model, X_base, y, random_state=random_state, **kwargs)


def create_model_grid_search_cv(model, dataset, x_cols, baseline_x_cols, random_state=10, **kwargs):

    X = dataset[x_cols].sort_index(axis=1)
    y = dataset['label'].values.ravel()

    gs = GridSearchCV(estimator=model(), param_grid=kwargs, scoring='roc_auc', cv=2)

    y_pred_vali = cross_val_predict(gs, X, y, cv=4, method='predict_proba')[:, 1]

    X_base = dataset[baseline_x_cols].sort_index(axis=1)

    _, y_pred_vali_base = create_model_from_dataset(model, X_base, y, random_state=random_state)


if __name__ == '__main__':
    search_params = {}

    # primary model
    BASE_MODEL_X_COLS = ['gender', 'age_group']
    X_COLS = BASE_MODEL_X_COLS + \
             ['symptom_well',
              'symptom_sore_throat',
              'symptom_cough',
              'symptom_shortness_of_breath',
              'symptom_smell_or_taste_loss',
              'symptom_fever',
              'condition_any']
    data = pd.read_csv('simulated_data.csv')
    if model == XGBClassifier:
        create_model_grid_search_cv(model=model, dataset=data, x_cols=X_COLS, baseline_x_cols=BASE_MODEL_X_COLS, **search_params)
    else:
        create_model_without_validation(model=model, dataset=data, x_cols=X_COLS, baseline_x_cols=BASE_MODEL_X_COLS)

    # extended model
    BASE_MODEL_X_COLS = ['gender', 'age_group']
    X_COLS = BASE_MODEL_X_COLS + \
             ['symptom_well',
              'symptom_shortness_of_breath',
              'symptom_runny_nose',
              'symptom_fatigue',
              'symptom_nausea_vomiting',
              'symptom_muscle_pain',
              'symptom_sore_throat',
              'symptom_cough_dry',
              'symptom_cough_moist',
              'symptom_diarrhea',
              'symptom_fever',
              'symptom_chills',
              'symptom_confusion',
              'symptom_smell_or_taste_loss',
              'condition_diabetes',
              'condition_hypertention',
              'condition_ischemic_heart_disease',
              'condition_lung_disease',
              'condition_kidney_disease',
              'condition_cancer',
              'condition_immune_system_suppression']
    data = pd.read_csv('simulated_data.csv')
    if model == XGBClassifier:
        create_model_grid_search_cv(model=model, dataset=data, x_cols=X_COLS, baseline_x_cols=BASE_MODEL_X_COLS, **search_params)
    else:
        create_model_without_validation(model=model, dataset=data, x_cols=X_COLS, baseline_x_cols=BASE_MODEL_X_COLS)

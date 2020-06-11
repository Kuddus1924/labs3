import numpy as np
import function

from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier


def main():
    grad_boosting = GradientBoostingClassifier()
    liner_svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=False)
    paramsBoost = {
        'loss': ['exponential'],
        'learning_rate': np.linspace(0.01, 1.0, 10),
        'max_depth': [3,4,5]
    }
    paramsSVC = {
        'C': [10 ** p for p in range(-7, 8)]
    }
    grid_cv_boost, prediction_boost = function.learning_best_classifier(
        'data/train.csv',
        'data/test.csv',
        grad_boosting,
        paramsBoost,
        encode_columns=[1, 3, 5, 6, 7, 8, 9, 13],
        n_jobs=-1)
    grid_cv_liner, prediction_liner = function.learning_best_classifier(
        'data/train.csv',
        'data/test.csv',
        liner_svc,
        paramsSVC,
        encode_columns=[1, 3, 5, 6, 7, 8, 9, 13],
        n_jobs=-1)
    prediction_filename = f'data/gradBoosting{grid_cv_boost.best_score_:.4f}'
    function.save(prediction_filename, prediction_boost, grid_cv_boost, fmt='%i')
    prediction_filename = f'data/linearSVC{grid_cv_liner.best_score_:.4f}'
    function.save(prediction_filename, prediction_liner, grid_cv_liner, fmt='%i')


if __name__ == '__main__':
    main()

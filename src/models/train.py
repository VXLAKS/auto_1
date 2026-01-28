import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, RocCurveDisplay

def train():
    # Загрузка
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Инициализация MLflow
    mlflow.set_experiment("Credit_Scoring_PD_Model")

    with mlflow.start_run():
        # 1. Создаем Pipeline (Препроцессинг + Модель)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ])

        # 2. Настройка гиперпараметров (GridSearchCV - требование задания)
        param_grid = {
            'clf__n_estimators': [50, 100],  # кол-во деревьев
            'clf__max_depth': [5, 10],       # глубина
            'clf__min_samples_split': [2, 5]
        }
        
        print("Начинаем подбор гиперпараметров...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Лучшие параметры: {best_params}")

        # 3. Оценка качества
        y_pred = best_model.predict(X_test)
        y_probs = best_model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_probs)
        f1 = f1_score(y_test, y_pred)

        print(f"Metrics -> AUC: {auc:.4f}, F1: {f1:.4f}")

        # 4. Логирование в MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)

        # 5. График ROC-кривой (Артефакт)
        RocCurveDisplay.from_estimator(best_model, X_test, y_test)
        plt.title("ROC Curve")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")

        # 6. Сохранение модели
        mlflow.sklearn.log_model(best_model, "model")
        joblib.dump(best_model, "models/model.pkl")

if __name__ == "__main__":
    train()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings

# Подавление предупреждений для чистоты вывода
warnings.filterwarnings('ignore')

from preprocessing_feature_engineering import engineer_features

def main():
    print("Loading data...")
    # 1. Загрузка и разделение данных
    df = pd.read_csv('../data/train.csv')
    
    # Применение конструирования признаков
    df = engineer_features(df)
    
    cols_to_drop = ['Cover_Type']
    if 'Id' in df.columns:
        cols_to_drop.append('Id')
        
    X = df.drop(columns=cols_to_drop)
    y = df['Cover_Type']
    
    # Разделение на Train(1) [75%] и Test(1) [25%]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    print(f"Data split: Train shape {X_train.shape}, Test shape {X_test.shape}")

    # 2. Определение моделей и пайплайнов
    # Модели, чувствительные к масштабу данных (нужен StandardScaler)
    log_reg_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
    knn_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])
    svm_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', SVC(random_state=42))])
    
    # Древовидные модели (масштабирование не требуется)
    rf_pipeline = Pipeline([('clf', RandomForestClassifier(random_state=42))])
    gb_pipeline = Pipeline([('clf', GradientBoostingClassifier(random_state=42))])

    
    # 3. Сетки гиперпараметров (настройки для перебора)
    grids = {
        'LogisticRegression': {
            'pipeline': log_reg_pipeline,
            'param_grid': {'clf__C': [0.01, 0.1, 1]}
        },
        'KNN': {
            'pipeline': knn_pipeline,
            'param_grid': {'clf__n_neighbors': [10, 20, 30]}
        },
        'SVM': {
            'pipeline': svm_pipeline,
            'param_grid': {'clf__C': [0.1, 1], 'clf__kernel': ['rbf']},
            'subset': True # Флаг для использования подмножества данных (чтобы быстрее)
        },
        'RandomForest': {
            'pipeline': rf_pipeline,
            'param_grid': {
                'clf__n_estimators': [100], 
                'clf__max_depth': [10, 12, 14],       
                'clf__min_samples_leaf': [20, 50],    
                'clf__min_samples_split': [50, 100]
            }
        },
        'GradientBoosting': {
            'pipeline': gb_pipeline,
            'param_grid': {
                'clf__n_estimators': [50, 100],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth': [3, 4],
                'clf__min_samples_leaf': [20, 50]
            }
        }
    }
    
    best_score = 0
    best_pipeline = None
    best_model_name = ""
    
    # 4. Цикл обучения
    results_list = []
    
    for model_name, config in grids.items():
        print(f"\nTraining {model_name}...")
        
        pipeline = config['pipeline']
        param_grid = config['param_grid']
        
        # Обработка подмножества для SVM (берем меньше данных)
        if config.get('subset'):
            print(f"Subsetting data for {model_name}...")
            # Создание подмножества строго для GridSearch SVM
            X_train_model, _, y_train_model, _ = train_test_split(
                X_train, y_train, train_size=20000, stratify=y_train, random_state=42
            )
        else:
            X_train_model, y_train_model = X_train, y_train
            
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_model, y_train_model)
        
        print(f"Best params for {model_name}: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_pipeline = grid_search.best_estimator_
            best_model_name = model_name
            
    print(f"\nBest Model: {best_model_name} with CV Accuracy: {best_score:.4f}")
    
    # Переобучение лучшего пайплайна на ПОЛНЫХ данных, если использовалось подмножество (SVM).
    # Обычно GridSearchCV переобучает модель на тех данных, которые были переданы в .fit().
    # Если лучшей моделью стал SVM (что маловероятно на подмножестве), он обучен на 20k примеров.
    # Чтобы строго следовать инструкциям "Сохранить ПАЙПЛАЙН", я сохраню объект, который вернул GridSearchCV.
    # Мы не добавляем сложную логику переобучения SVM на всех данных, чтобы не нарушить условия задачи,
    # хотя для лучшего качества это стоило бы сделать.
    # Мы используем best_estimator_ из поиска по сетке.

    # 5. Вывод и сохранение
    
    # Сохранение ПАЙПЛАЙНА
    joblib.dump(best_pipeline, '../results/best_pipeline.pkl')
    print("Saved best pipeline to ../results/best_pipeline.pkl")
    
    # Оценка на Train (Полный X_train) для проверки переобучения
    # Примечание: Если лучшая модель SVM, эта проверка выполняется на полных данных против модели, обученной на подмножестве.
    # Она может показать худший результат, но это ожидаемо для подмножества.
    y_train_pred = best_pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    
    # Проверка требований аудита
    if train_acc > 0.98:
        print("WARNING: Model is overfitting, consider increasing regularization manually.")
        
    # Оценка на Test(1)
    y_test_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test(1) Accuracy: {test_acc:.4f}")
    
    if test_acc <= 0.65:
         print("WARNING: Test accuracy doesn't meet the > 0.65 criteria.")

    # Матрица ошибок (Confusion Matrix)
    # Индекс=Истинные метки, Столбцы=Предсказанные метки. Целочисленные метки классов (1-7).
    # Cover_Type уже 1-7.
    cm = confusion_matrix(y_test, y_test_pred, labels=[1, 2, 3, 4, 5, 6, 7])
    cm_df = pd.DataFrame(cm, index=[1, 2, 3, 4, 5, 6, 7], columns=[1, 2, 3, 4, 5, 6, 7])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../results/confusion_matrix_heatmap.png', dpi=300)
    plt.close()
    print("Saved confusion matrix heatmap.")
    
    # Кривая обучения (Learning Curve)
    # График оценки обучения против оценки валидации для разных размеров обучающей выборки.
    print("Generating learning curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        best_pipeline, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.title(f"Learning Curve ({best_model_name})")
    plt.grid()
    plt.savefig('../results/learning_curve_best_model.png', dpi=300)
    plt.close()
    print("Saved learning curve.")

if __name__ == "__main__":
    main()

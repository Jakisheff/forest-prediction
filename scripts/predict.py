import pandas as pd
import joblib
from preprocessing_feature_engineering import engineer_features

def main():
    try:
        # 1. Загрузка лучшего пайплайна (обученной модели)
        pipeline = joblib.load('../results/best_pipeline.pkl')
        print("Loaded pipeline from ../results/best_pipeline.pkl")
    except FileNotFoundError:
        print("Error: ../results/best_pipeline.pkl not found. Please run model_selection.py first.")
        return

    # 2. Загрузка тестовых данных
    try:
        df_test = pd.read_csv('../data/test.csv')
    except FileNotFoundError:
        print("Error: ../data/test.csv not found.")
        return

    # 2.5 Обработка колонок Id и Cover_Type
    if 'Id' in df_test.columns:
        test_ids = df_test['Id']
        if 'Cover_Type' in df_test.columns:
             df_test = df_test.drop(columns=['Cover_Type'])
    else:
        # Генерация Id, если его нет (предполагаем нумерацию с 1)
        test_ids = range(1, len(df_test) + 1)
        if 'Cover_Type' in df_test.columns:
             df_test = df_test.drop(columns=['Cover_Type'])

    # 3. Применение конструирования признаков (Feature Engineering)
    # Примечание: engineer_features может обрабатывать 'Id' или другие колонки?
    # Функция возвращает таблицу с новыми признаками и удаляет исходные.
    # Нам нужно убедиться, что мы не удаляем столбцы, которые ожидает модель,
    # и что у нас нет лишних столбцов.
    # Скрипт обучения удаляет 'Cover_Type' и 'Id'.
    # Здесь мы тоже должны подать в модель правильные столбцы.
    
    df_test_engineered = engineer_features(df_test)
    
    # Удаляем 'Id' перед предсказанием, так как он был удален при обучении
    if 'Id' in df_test_engineered.columns:
        X_test = df_test_engineered.drop(columns=['Id'])
    else:
        X_test = df_test_engineered

    # 4. Предсказание
    print("Generating predictions...")
    y_pred = pipeline.predict(X_test)

    # 5. Сохранение результатов
    submission = pd.DataFrame({
        'Id': test_ids,
        'Cover_Type': y_pred
    })

    submission.to_csv('../results/test_predictions.csv', index=False)
    print("Saved predictions to ../results/test_predictions.csv")

if __name__ == "__main__":
    main()

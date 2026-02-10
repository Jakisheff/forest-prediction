import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Обрабатывает создание признаков (feature engineering) для train и test без утечки данных.
    
    Логика:
    1. Soil Type Mapping: Обратное One-Hot кодирование 40 колонок Soil_Type -> 1 Soil_Type (1-40).
       Отображение в Climatic_Zone и Geologic_Zone.
    2. Wilderness Area Mapping: Обратное One-Hot кодирование Wilderness_Area1-4 -> Wilderness_ID (1-4).
    3. Feature Engineering: Euclidean_Dist_Hydro, Sin_Aspect, Cos_Aspect, Mean_Hillshade, 
       Hillshade_Contrast, Interaction_Fire_Road.
    """
    df = df.copy()
    
    # --- 1. Обработка типов почвы (Soil Type) ---
    # Обратное One-Hot кодирование Soil_Type (из 40 колонок делаем одну)
    soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]
    # Используем idxmax, чтобы найти столбец со значением 1, затем извлекаем число.
    # Если все 0 (чего не должно быть в валидных данных), это может потребовать обработки.
    # Предполагаем валидные данные, где ровно одна единица.
    df['Soil_Type'] = df[soil_cols].idxmax(axis=1).apply(lambda x: int(x.replace('Soil_Type', '')))
    
    # Удаляем исходные бинарные столбцы
    df.drop(columns=soil_cols, inplace=True)
    
    # Отображение в Climatic_Zone (Климатическая зона) и Geologic_Zone (Геологическая зона)
    SOIL_MAPPING = {
        1: {'climatic_zone': 2, 'geologic_zone': 7}, 2: {'climatic_zone': 2, 'geologic_zone': 7},
        3: {'climatic_zone': 2, 'geologic_zone': 7}, 4: {'climatic_zone': 2, 'geologic_zone': 7},
        5: {'climatic_zone': 2, 'geologic_zone': 7}, 6: {'climatic_zone': 2, 'geologic_zone': 7},
        7: {'climatic_zone': 3, 'geologic_zone': 5}, 8: {'climatic_zone': 3, 'geologic_zone': 5},
        9: {'climatic_zone': 4, 'geologic_zone': 2}, 10: {'climatic_zone': 4, 'geologic_zone': 7},
        11: {'climatic_zone': 4, 'geologic_zone': 7}, 12: {'climatic_zone': 4, 'geologic_zone': 7},
        13: {'climatic_zone': 4, 'geologic_zone': 7}, 14: {'climatic_zone': 5, 'geologic_zone': 1},
        15: {'climatic_zone': 5, 'geologic_zone': 1}, 16: {'climatic_zone': 6, 'geologic_zone': 1},
        17: {'climatic_zone': 6, 'geologic_zone': 1}, 18: {'climatic_zone': 6, 'geologic_zone': 7},
        19: {'climatic_zone': 7, 'geologic_zone': 1}, 20: {'climatic_zone': 7, 'geologic_zone': 1},
        21: {'climatic_zone': 7, 'geologic_zone': 1}, 22: {'climatic_zone': 7, 'geologic_zone': 2},
        23: {'climatic_zone': 7, 'geologic_zone': 2}, 24: {'climatic_zone': 7, 'geologic_zone': 7},
        25: {'climatic_zone': 7, 'geologic_zone': 7}, 26: {'climatic_zone': 7, 'geologic_zone': 7},
        27: {'climatic_zone': 7, 'geologic_zone': 7}, 28: {'climatic_zone': 7, 'geologic_zone': 7},
        29: {'climatic_zone': 7, 'geologic_zone': 7}, 30: {'climatic_zone': 7, 'geologic_zone': 7},
        31: {'climatic_zone': 7, 'geologic_zone': 7}, 32: {'climatic_zone': 7, 'geologic_zone': 7},
        33: {'climatic_zone': 7, 'geologic_zone': 7}, 34: {'climatic_zone': 7, 'geologic_zone': 7},
        35: {'climatic_zone': 8, 'geologic_zone': 7}, 36: {'climatic_zone': 8, 'geologic_zone': 7},
        37: {'climatic_zone': 8, 'geologic_zone': 7}, 38: {'climatic_zone': 8, 'geologic_zone': 7},
        39: {'climatic_zone': 8, 'geologic_zone': 7}, 40: {'climatic_zone': 8, 'geologic_zone': 7}
    }
    
    df['Climatic_Zone'] = df['Soil_Type'].map(lambda x: SOIL_MAPPING[x]['climatic_zone'])
    df['Geologic_Zone'] = df['Soil_Type'].map(lambda x: SOIL_MAPPING[x]['geologic_zone'])
    
    # --- 2. Обработка природных зон (Wilderness Area) ---
    # Обратное One-Hot кодирование Wilderness_Area (из 4 колонок делаем одну)
    wild_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
    df['Wilderness_ID'] = df[wild_cols].idxmax(axis=1).apply(lambda x: int(x.replace('Wilderness_Area', '')))
    
    # Удаляем исходные бинарные столбцы
    df.drop(columns=wild_cols, inplace=True)
    
    # --- 3. Создание новых признаков (Feature Engineering) ---
    # Евклидово расстояние до воды (по прямой)
    df['Euclidean_Dist_Hydro'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)
    
    # Sin_Aspect / Cos_Aspect (Синус и Косинус аспекта)
    # Aspect в градусах, переводим в радианы
    df['Sin_Aspect'] = np.sin(df['Aspect'] * np.pi / 180)
    df['Cos_Aspect'] = np.cos(df['Aspect'] * np.pi / 180)
    
    # Mean_Hillshade (Средняя освещенность)
    df['Mean_Hillshade'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3
    
    # Hillshade_Contrast (Контраст освещенности)
    df['Hillshade_Contrast'] = np.abs(df['Hillshade_9am'] - df['Hillshade_3pm'])
    
    # Interaction_Fire_Road (Разница расстояний до огня и дорог)
    df['Interaction_Fire_Road'] = df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways']
    
    return df

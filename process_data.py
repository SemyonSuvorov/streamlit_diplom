import os
import pandas as pd

# Получаем абсолютный путь к текущей директории
current_dir = os.path.dirname(os.path.abspath(__file__))
# Формируем путь к файлу
file_path = os.path.join(current_dir, 'data', 'forecast_xgboost.csv')
pred_data = pd.read_csv(file_path)

# Удаляем столбец 'date'
pred_data = pred_data.drop('date', axis=1)

# Переименовываем столбец 'Value' в 'Sales'
pred_data = pred_data.rename(columns={'Value': 'Sales'})

# Создаем новый индекс, начинающийся с 1
pred_data.index = range(1, len(pred_data) + 1)
pred_data.index.name = 'Id'

# Сохраняем результат
pred_data.to_csv(file_path) 
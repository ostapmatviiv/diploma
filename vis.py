import pandas as pd
import matplotlib.pyplot as plt

# Завантаження датасету
dataset = pd.read_csv('epoch_reward10.csv')

# Вибір двох колонок для візуалізації
column1 = dataset['Epoch']
column2 = dataset['Reward']

# Візуалізація двох колонок
# plt.plot(column1, label='Epoch')
plt.plot(column2, label='Reward')

# Налаштування осей та заголовку графіку
plt.xlabel('Epoch-axis')
plt.ylabel('Reward-axis')
# plt.title('Visualization of Column 1 and Column 2')

# Додавання легенди
plt.legend()

# Показ графіку
plt.show()

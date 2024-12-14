#импортируем нужные библиотеки
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('Titanic.csv')

print(data)

data['Age'] = data['Age'].fillna(data['Age'].median())  # пропуски в возрастной ячейке заполняем медианным значением

# строим график выживаемости
df = data['Survived'].value_counts()
x = ['Погиб', 'Выжил']
y = df.iloc[[0, 1]]
plt.bar(x, y, color=['red', 'blue'])
plt.title('График выживаемости')
plt.xlabel('Количество людей')
plt.ylabel('Выживаемость')
plt.show()

# график выживаемости по полу
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('График выживаемости по полу')
plt.xlabel('Пол')
plt.ylabel('Количество людей')
plt.legend(title='Выживаемость', labels=['Погиб', 'Выжил'])
plt.show()

# график выживаемости по классу обслуживания
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('График выживаемости по классу')
plt.xlabel('Класс')
plt.ylabel('Количество людей')
plt.legend(title='Выживаемость', labels=['Погиб', 'Выжил'])
plt.show()

plt.hist([data[data['Survived'] == 1]['Age'], data[data['Survived'] == 0]['Age']], bins=30, color=['green', 'red'], label=['Выжил', 'Погиб'])
plt.xlabel('Возраст')
plt.ylabel('Количество пассажиров')
plt.title('График выживаемости по возрасту')
plt.legend()
plt.show()

plt.hist([data[data['Sex'] == 'male']['Age'], data[data['Sex'] == 'female']['Age']], bins=30, color=['green', 'red'], label=['Мужчина', 'Женщина'])
plt.xlabel('Возраст')
plt.ylabel('Количество пассажиров')
plt.title('График количества людей разного пола и возраста')
plt.legend()
plt.show()

# для семьи (SibSp - число братьев, супругов и Parch - дети и родители
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1  # размер семьи

data['Is_Alone'] = (data['Family_Size'] == 1).astype(int)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Family_Size', 'Is_Alone']
X = pd.get_dummies(data[features], drop_first=True)  # преобразуем категориальные данные
y = data['Survived']  # целевая переменная

X.fillna(0, inplace=True)  # убираем пропуски
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # масштабируем тренировочные данные
X_test = scaler.transform(X_test)  # масштабируем тестовые данные

model = RandomForestClassifier(n_estimators=100, random_state=42)  # создаем модель случайного леса
model.fit(X_train, y_train)  # обучаем модель

y_pred = model.predict(X_test)  # делаем предсказания

accuracy = accuracy_score(y_test, y_pred)  # точность модели
conf_matrix = confusion_matrix(y_test, y_pred)  # матрица ошибок
class_report = classification_report(y_test, y_pred)  # отчет по классификации

print(f"Точность модели: {accuracy}")  # вывод точности
print("Классификация:")
print(class_report)  # классификация

# Важность признаков
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
print("Важность признаков:")
print(feature_importances)  # важность каждого признака

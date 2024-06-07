import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Загрузка данных
data = pd.read_csv('heart_failure_clinical_records.csv')

# Разделение данных на признаки и целевую переменную
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Создание интерфейса для изменения гиперпараметров
st.title("Random Forest")

st.sidebar.header('Изменение гиперпараметров:')
max_depth = st.sidebar.slider("Глубина дерева (max_depth):", 1, 10, 5)
n_estimators = st.sidebar.slider("Количество деревьев (n_estimators):", 10, 1000, 100)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Предсказание и оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

# Вывод результатов
st.write("Accuracy:", accuracy)
st.write("Матрица ошибок:")
st.write(conf_mat)

# Вывод графика ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(conf_mat[1], conf_mat[0], label='Random Forest')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend()
st.pyplot(plt)

# Вывод графика важности признаков
feature_importances = model.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(range(X.shape[1]), feature_importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.yticks(range(X.shape[1]), X.columns)  # Добавление названий признаков
st.pyplot(plt)

# Вывод графика классификационной отчетности
plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, interpolation='nearest')
plt.title('Тепловая карта матрицы ошибок')
plt.colorbar()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
st.pyplot(plt)

# Вывод графика Precision-Recall
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
st.pyplot(plt)
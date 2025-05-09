import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import random

# --- Model & Hyperparameter Options ---
MODEL_SPACE = {
    'LogisticRegression': {
        'model': LogisticRegression,
        'params': {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 200]}
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier,
        'params': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    },
    'SVC': {
        'model': SVC,
        'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier,
        'params': {'max_depth': [3, 5, 10, None]}
    },
    'GaussianNB': {
        'model': GaussianNB,
        'params': {}
    }
}

def random_individual():
    model_name = random.choice(list(MODEL_SPACE.keys()))
    model_info = MODEL_SPACE[model_name]
    params = {k: random.choice(v) for k, v in model_info['params'].items()}
    return (model_name, params)

def mutate(individual):
    model_name, params = individual
    model_info = MODEL_SPACE[model_name]
    if random.random() < 0.3:
        model_name = random.choice(list(MODEL_SPACE.keys()))
        params = {k: random.choice(v) for k, v in MODEL_SPACE[model_name]['params'].items()}
    else:
        for k in params:
            if random.random() < 0.5:
                params[k] = random.choice(model_info['params'][k])
    return (model_name, params)

def crossover(ind1, ind2):
    return ind1 if random.random() > 0.5 else ind2

def fitness(individual, X_train, X_test, y_train, y_test):
    model_name, params = individual
    Model = MODEL_SPACE[model_name]['model']
    try:
        model = Model(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)
    except:
        return 0

def genetic_algorithm(X_train, X_test, y_train, y_test, generations=10, population_size=10):
    population = [random_individual() for _ in range(population_size)]
    for _ in range(generations):
        scored = [(ind, fitness(ind, X_train, X_test, y_train, y_test)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        population = [x[0] for x in scored[:population_size // 2]]
        while len(population) < population_size:
            p1, p2 = random.sample(population, 2)
            child = mutate(crossover(p1, p2))
            population.append(child)
    best = max(population, key=lambda ind: fitness(ind, X_train, X_test, y_train, y_test))
    return best

# --- Streamlit UI ---
st.title("AutoML with Genetic Algorithm")
st.write("Upload a CSV file to automatically find the best model and hyperparameters.")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Preview of Data:")
    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    if st.button("Run AutoML"):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        if y.dtype == 'O':
            y = LabelEncoder().fit_transform(y)

        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.info("Running Genetic Algorithm for model selection...")
        best_model, best_params = genetic_algorithm(X_train, X_test, y_train, y_test)

        st.success(f"Best Model: {best_model}")
        st.json(best_params)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import chardet
from PIL import Image
import io

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
    file_path = 'kimchi_data.csv'
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        df = pd.read_csv(file_path, encoding=encoding)
        st.success(f"Successfully loaded data with {encoding} encoding.")
        
        # 열 이름 출력
        st.write("Original column names:", df.columns.tolist())
        st.write(f"Number of columns: {len(df.columns)}")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_model(X, y, test_size, k_neighbors):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=77)
    model = KNeighborsRegressor(n_neighbors=k_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, X_test

def plot_correlation(data, features, target):
    corr_data = data[features + [target]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={"shrink": .8})
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return plt

def main():
    st.title("Machine Learning Application for Kimchi Ingredient Prediction")

    kimchi_data = load_data()
    if kimchi_data is None:
        st.stop()

    st.write("Kimchi dataset preview:", kimchi_data.head())

    # 데이터 전처리
    numeric_columns = kimchi_data.select_dtypes(include=['float64', 'int64']).columns
    kimchi_data = kimchi_data[numeric_columns]

    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Data Ratio', 0.1, 0.5, 0.2)
    k_neighbors = st.sidebar.slider('Number of Neighbors for K-NN Model', 1, 20, 5)

    input_features = kimchi_data.columns[:-5].tolist()  # 마지막 5개 열을 제외한 모든 열
    selected_features = st.multiselect("Select features for prediction", input_features, default=input_features[:3])

    target_options = kimchi_data.columns[-5:].tolist()  # 마지막 5개 열
    target_column = st.selectbox("Select target variable for prediction", target_options)

    if not selected_features or not target_column:
        st.warning("Please select both features and target variable.")
        st.stop()

    X = kimchi_data[selected_features]
    y = kimchi_data[target_column]

    try:
        model, mse, r2, X_test = train_model(X, y, test_size, k_neighbors)

        st.write(f"Model Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Coefficient of Determination (R^2): {r2:.2f}")

        st.sidebar.header('New Data Prediction')
        user_input = {feature: st.sidebar.number_input(f'{feature}', value=X[feature].mean()) for feature in selected_features}

        if st.sidebar.button('Run Prediction'):
            prediction = model.predict(pd.DataFrame([user_input]))
            st.sidebar.write(f"Predicted {target_column}: {prediction[0]:.2f}")

        st.header("Data Visualization")
        st.write("Correlation between selected features and target variable:")
        fig = plot_correlation(kimchi_data, selected_features, target_column)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error occurred during model training or prediction: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import chardet

@st.cache_data
def load_data():
    file_path = 'kimchi_data.csv'  # 파일 경로 확인
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
    
    st.write(f"Detected file encoding: {detected_encoding}")
    
    encodings = [detected_encoding, 'utf-8', 'cp949', 'euc-kr']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if df.empty:
                st.warning(f"File read with {encoding} encoding, but data is empty.")
                continue
            st.success(f"Successfully loaded data with {encoding} encoding.")
            # 컬럼명을 영어로 변경
            df.columns = ['year', 'month', 'avg_temp', 'avg_max_temp', 'max_temp', 'avg_min_temp', 'min_temp', 
                          'avg_monthly_rain', 'max_monthly_rain', 'max_hourly_rain',
                          'cabbage_price', 'radish_price', 'red_pepper_price', 'garlic_price', 'green_onion_price']
            return df
        except UnicodeDecodeError:
            st.warning(f"Failed to read with {encoding} encoding")
        except Exception as e:
            st.error(f"Unexpected error while loading data: {str(e)}")
    
    st.error("All encoding attempts failed. Please check the file format and content.")
    return None

def train_model(X, y, test_size, k_neighbors):
    if X.empty or y.empty:
        st.error("Feature or target data is empty.")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=77)
    model = KNeighborsRegressor(n_neighbors=k_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, X_test

def plot_correlation(data, features, target):
    corr_data = data[features + [target]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax, fmt=".2f", cbar_kws={"shrink": .8})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def main():
    st.title("Machine Learning Application for Kimchi Ingredient Prediction")

    kimchi_data = load_data()
    if kimchi_data is None:
        st.stop()

    st.write("Dataframe columns:", kimchi_data.columns.tolist())
    st.write("Kimchi dataset preview:", kimchi_data.head())

    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Data Ratio', 0.1, 0.5, 0.2)
    k_neighbors = st.sidebar.slider('Number of Neighbors for K-NN Model', 1, 20, 5)

    input_features = ['avg_temp', 'avg_max_temp', 'max_temp', 'avg_min_temp', 'min_temp', 
                      'avg_monthly_rain', 'max_monthly_rain', 'max_hourly_rain']
    selected_features = st.multiselect("Select features for prediction", input_features, default=input_features[:3])

    target_options = ['cabbage_price', 'radish_price', 'red_pepper_price', 'garlic_price', 'green_onion_price']
    target_column = st.selectbox("Select target variable for prediction", target_options)

    X = kimchi_data[selected_features]
    y = kimchi_data[target_column]

    if X.empty or y.empty:
        st.error("Feature or target data is empty.")
        st.stop()

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

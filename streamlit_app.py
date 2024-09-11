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
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
    
    st.write(f"감지된 파일 인코딩: {detected_encoding}")
    
    encodings = [detected_encoding, 'utf-8', 'cp949', 'euc-kr']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if df.empty:
                st.warning(f"{encoding} 인코딩으로 파일을 읽었으나 데이터가 비어있습니다.")
                continue
            st.success(f"성공적으로 {encoding} 인코딩으로 데이터를 로드했습니다.")
            return df
        except UnicodeDecodeError:
            st.warning(f"{encoding} 인코딩으로 읽기 실패")
        except Exception as e:
            st.error(f"데이터 로드 중 예상치 못한 오류 발생: {str(e)}")
    
    st.error("모든 인코딩 시도가 실패했습니다. 파일 형식과 내용을 확인해주세요.")
    return None

def train_model(X, y, test_size, k_neighbors):
    if X.empty or y.empty:
        st.error("특성 또는 타겟 데이터가 비어있습니다.")
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
    
    plt.xticks(rotation=45, ha='right', fontsize=10)

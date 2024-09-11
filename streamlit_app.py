import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import chardet
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_manager.fontManager.addfont(font_path)
rc('font', family='Malgun Gothic')

@st.cache_data
def load_data():
    file_path = 'data/kimchi_data.csv'
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
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
    plt.xticks(rotation=45)  # x축 레이블 회전
    plt.yticks(rotation=45)  # y축 레이블 회전
    plt.tight_layout()  # 레이아웃 조정
    return fig

def main():
    st.title("김장 재료 예측을 위한 머신러닝 애플리케이션")

    kimchi_data = load_data()
    if kimchi_data is None:
        st.stop()

    st.write("데이터프레임 열:", kimchi_data.columns.tolist())
    st.write("김장 데이터셋 미리보기:", kimchi_data.head())

    st.sidebar.header('모델 매개변수')
    test_size = st.sidebar.slider('테스트 데이터 비율', 0.1, 0.5, 0.2)
    k_neighbors = st.sidebar.slider('K-NN 모델의 이웃 개수', 1, 20, 5)

    input_features = ['평균기온', '평균최고기온', '최고기온', '평균최저기온', '최저기온', 
                      '평균월강수량', '최다월강수량', '1시간최다강수량']
    selected_features = st.multiselect("예측에 사용할 특성을 선택하세요", input_features, default=input_features[:3])

    target_options = ['배추값', '무값', '고추값', '마늘값', '쪽파값']
    target_column = st.selectbox("예측할 타겟 변수를 선택하세요", target_options)

    X = kimchi_data[selected_features]
    y = kimchi_data[target_column]

    if X.empty or y.empty:
        st.error("특성 또는 타겟 데이터가 비어있습니다.")
        st.stop()

    try:
        model, mse, r2, X_test = train_model(X, y, test_size, k_neighbors)

        st.write(f"모델 평균 제곱 오차 (MSE): {mse:.2f}")
        st.write(f"결정 계수 (R^2): {r2:.2f}")

        st.sidebar.header('새 데이터 예측')
        user_input = {feature: st.sidebar.number_input(f'{feature}', value=X[feature].mean()) for feature in selected_features}

        if st.sidebar.button('예측 실행'):
            prediction = model.predict(pd.DataFrame([user_input]))
            st.sidebar.write(f"예측된 {target_column}: {prediction[0]:.2f}")

        st.header("데이터 시각화")
        st.write("선택된 특성과 타겟 변수 간의 상관관계:")
        fig = plot_correlation(kimchi_data, selected_features, target_column)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"모델 훈련 또는 예측 중 오류 발생: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()

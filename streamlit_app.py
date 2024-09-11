def load_data():
    file_path = 'kimchi_data.csv'
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
            
            # 열 이름 출력
            st.write("Original column names:", df.columns.tolist())
            st.write(f"Number of columns: {len(df.columns)}")
            
            # 열 이름을 영어로 변경 (실제 열 개수에 맞게 조정)
            english_columns = ['year', 'month', 'avg_temp', 'avg_max_temp', 'max_temp', 'avg_min_temp', 'min_temp',
                               'avg_monthly_rain', 'max_monthly_rain', 'max_hourly_rain']
            
            # 나머지 열은 자동으로 이름 생성
            for i in range(len(df.columns) - len(english_columns)):
                english_columns.append(f'column_{i+1}')
            
            df.columns = english_columns
            
            # 새로운 열 이름 출력
            st.write("New column names:", df.columns.tolist())
            
            return df
        except UnicodeDecodeError:
            st.warning(f"Failed to read with {encoding} encoding")
        except Exception as e:
            st.error(f"Unexpected error while loading data: {str(e)}")
    
    st.error("All encoding attempts failed. Please check the file format and content.")
    return None

# main() 함수 내부의 관련 부분 수정
def main():
    # ... (이전 코드)
    
    kimchi_data = load_data()
    if kimchi_data is None:
        st.stop()

    st.write("Kimchi dataset preview:", kimchi_data.head())

    # 동적으로 입력 특성과 타겟 변수 선택
    all_columns = kimchi_data.columns.tolist()[2:]  # 연도와 월을 제외한 모든 열
    input_features = all_columns[:-5]  # 마지막 5개 열을 제외한 나머지를 입력 특성으로 사용
    selected_features = st.multiselect("Select features for prediction", input_features, default=input_features[:3])

    target_options = all_columns[-5:]  # 마지막 5개 열을 타겟 옵션으로 사용
    target_column = st.selectbox("Select target variable for prediction", target_options)

    # ... (나머지 코드)

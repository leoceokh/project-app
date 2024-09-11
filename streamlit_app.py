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
            
            # 데이터프레임의 열 이름과 개수 출력
            st.write(f"Columns in the dataframe: {df.columns.tolist()}")
            st.write(f"Number of columns: {len(df.columns)}")
            
            # 열 이름을 영어로 변경 (실제 데이터 구조에 맞게 수정 필요)
            new_columns = ['year', 'month', 'avg_temp', 'avg_max_temp', 'max_temp', 'avg_min_temp', 'min_temp', 
                           'avg_monthly_rain', 'max_monthly_rain', 'max_hourly_rain',
                           'cabbage_price', 'radish_price', 'red_pepper_price', 'garlic_price', 'green_onion_price']
            
            # 열 개수가 일치하는지 확인
            if len(df.columns) == len(new_columns):
                df.columns = new_columns
            else:
                st.warning(f"Column count mismatch. Expected {len(new_columns)}, got {len(df.columns)}.")
                st.write("Original column names:", df.columns.tolist())
                # 열 개수가 일치하지 않으면 원래 열 이름을 유지
            
            return df
        except UnicodeDecodeError:
            st.warning(f"Failed to read with {encoding} encoding")
        except Exception as e:
            st.error(f"Unexpected error while loading data: {str(e)}")
    
    st.error("All encoding attempts failed. Please check the file format and content.")
    return None

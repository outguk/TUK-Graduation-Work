import pickle

# 데이터 분석용 함수
# annotations 출력 코드 
# 결과 확인용용


def load_and_print_all(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"파일: {file_path}")
    print(f"데이터 타입: {type(data)}")
    if isinstance(data, dict):
        print(f"딕셔너리 키: {data.keys()}")
        if 'annotations' in data:
            print(f"annotations 길이: {len(data['annotations'])}")
            for idx, annotation in enumerate(data['annotations']):
                print(f"\nannotation {idx + 1}: {annotation}")

output_data = load_and_print_all("output.pkl")
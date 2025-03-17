
import pandas as pd
from random import shuffle
from os import listdir
from tqdm import tqdm
import json
import re
import os


# 라벨링 데이터 가져오는 함수
def get_data(data, path):
    id, script_stt_txt, script_tag_txt, taglist, average_grade =[], [], [], [], []  # 빈 리스트 생성 (6개)
    for row in tqdm(data.itertuples(), total=data.shape[0]):  # 진행상황 확인을 위한 tqdm, data frame 값을 빠르게 가져오기 위한 itertuples
        file_str = path + row.file_name                       # 파일명

        with open(file_str, encoding="utf-8") as f:      # json파일 열기
            text = json.load(f)

        id.append(text['info']['id'])  # id
        script_stt_txt.append(text['script']['script_stt_txt'])   # 발화 내용(STT)
        script_tag_txt.append(text['script']['script_tag_txt'])    # 발화 내용(태그매핑)
        taglist.append(text['taglist'])   # 태그정보
        average_grade.append(text['average']['eval_grade'])      # 총체적 평가  평균 등급


    df= pd.DataFrame({'id': id, 'script_stt_txt': script_stt_txt, 'script_tag_txt': script_tag_txt, 'taglist':taglist, 'average_grade': average_grade})  # 리스트로 dataframe 만들기
    return df


def label_text(text):
    labels = []
    words = text.split()
    
    for word in words:
        # 개체명 인식
        if  word.startswith("<") or word.endswith("/>"):
            labels.append(word)
        elif word.startswith("자") and word.endswith(">"):  # (예외:태그구문규칙 불일치) 자신의</REP><WR>이미지</WR><WR>강조하고</WR><WR>있는</WR> -> 자신의</REP> <WR>이미지</WR> <WR>강조하고</WR> <WR>있는</WR>
            labels.append(word)
        elif word == '다양성과<FIL>': # (예외:태그구문규칙 불일치) 다양성과<FIL> 포용을 </FIL>훼손하며 -> 다양성과 <FIL>포용을</FIL> 훼손하며
            labels.append(word)
        # 개체명이 아닌 경우 O 태그 부여
        else:
            labels.append('O')

    return labels

def merge_tags(text):
    text = re.sub(r'\*', '', text.strip()) # *
    text = re.sub(r'\*\w', 'O', text.strip()) # *O
    text = re.sub(r'<REP>\w\*\w', 'REP-B', text.strip())
    text = re.sub(r'<WR>\w+</WR><PS/><WR>\w+</WR>', 'WR-B PS-B WR-B', text.strip()) 
    text = re.sub(r'<PS/><WR>\w+</WR>', 'PS-B WR-B', text.strip()) 
    text = re.sub(r'<FIL>\w</FIL><PS/><FIL>\w</FIL>', 'FIL-B PS-B FIL-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR></REP><FIL>\w</FIL>', 'WR-B FIL-B', text.strip()) 
    text = re.sub(r'<PS/><FIL>\w</FIL>', 'PS-B FIL-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR><PS/><WR>\w+</WR>', 'WR-B PS-B WR-B', text.strip()) 
    text = re.sub(r'<REP>\w<FIL>\w</FIL>\w+</REP>', 'REP-B FIL-B O', text.strip()) 
    text = re.sub(r'<FIL>\w</FIL>\w+<FIL>\w</FIL><REP>\w', 'FIL-B O FIL-B REP-B', text.strip()) 
    text = re.sub(r'<REP>\w+<FIL>\w+</FIL>', 'REP-B FIL-B', text.strip()) 
    text = re.sub(r'<FIL>\w+</FIL><PS/><FIL>\w+</FIL>', 'FIL-B PS-B FIL-B ', text.strip()) 
    text = re.sub(r'<REP><WR>\w+</WR><FIL>\w+</FIL>', 'REP-B WR-B FIL-B ', text.strip()) 
    text = re.sub(r'<WR>\w+</WR></REP>\w+<WR></WR><FIL>\w+</FIL><REP>\w+', 'WR-B O WR-B FIL-B REP-B', text.strip()) 
    text = re.sub(r'<WR>\w+(-)\w+</WR>', 'WR-B', text.strip())
    text = re.sub(r'<REP><WR>\w+</WR><FIL>\w+</FIL>\w+</REP>', 'REP-B WR-B FIL-B O', text.strip()) 
    text = re.sub(r'<WR>\w+</WR></REP><WR>\w+</WR>', 'WR-B WR-B', text.strip()) 
    text = re.sub(r'<REP>\w+<WR>\w+</WR></REP><WR>\w+</WR><REP>\w+</REP>', 'REP-B WR-B WR-B REP-B', text.strip())
    text = re.sub(r'<REP>\w+<WR>\w+</WR></REP><WR>\w+</WR>', 'REP-B WR-B WR-B', text.strip()) 
    text = re.sub(r'<REP>\w+<WR>\w+</WR>\w+', 'REP-B WR-B O', text.strip()) 
    text = re.sub(r'<FIL>\w+</FIL><WR>\w+</WR></REP>', 'FIL-B WR-B', text.strip()) 
    text = re.sub(r'<FIL>\w+</FIL><WR>\w+</WR><REP>\w', 'FIL-B WR-B REP-B', text.strip()) 
    text = re.sub(r'<REP>\w+</REP><REP>\w+<WR>\w+</WR></REP>', 'REP-B REP-B WR-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR>\w+<WR>\w+</WR><REP>\w+', 'WR-B O WR-B REP-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR>\w+<WR>\w+</WR><WR>\w+</WR>', 'WR-B O WR-B WR-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR><FIL>\w+</FIL>\w+', 'WR-B FIL-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR><WR>\w+</WR>\w+<WR>\w+</WR>', 'WR-B WR-B O WR-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR></REP>\w+<WR></WR>', 'WR-B O WR-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR></REP>', 'WR-B O', text.strip()) 
    text = re.sub(r'<WR>\w+</WR><WR>\w+</WR><WR>\w+</WR><REP>\w+<WR>\w+</WR>', 'WR-B WR-B WR-B REP-B WR-B', text.strip())
    text = re.sub(r'<WR>\w+</WR><WR>\w+</WR><REP>\w+', 'WR-B WR-B REP-B', text.strip()) 
    text = re.sub(r'<REP><WR>\w+</WR>', 'REP-B WR-B', text.strip()) 
    text = re.sub(r'<REP>\w+<WR>\w+</WR>', 'REP-B WR-B', text.strip()) 
    text = re.sub(r'\w+</WR></REP> ', 'O', text.strip()) 
    text = re.sub(r'<REP>\w+<WR>\w+', 'REP-B WR-B', text.strip())
    text = re.sub(r'<PS/><WR></WR><FIL>\w+</FIL>', 'PS-B WR-B FIL-B', text.strip())
    text = re.sub(r'<WR><REP>\w+', 'WR-B REP-B', text.strip())
    text = re.sub(r'<WR>\w+</WR><PS/>', 'WR-B PS-B', text.strip())
    text = re.sub(r'<WR>\w+</WR><REP>\w+</REP><WR>\w+</WR>', 'WR-B REP-B WR-B', text.strip()) 
    text = re.sub(r'<FIL>\w+</FIL>\w+</REP>\w+<FIL>\w+</FIL><WR>\w+</WR>', 'FIL-B O FIL-B WR-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR><REP>\w', 'WR-B REP-B', text.strip()) 
    text = re.sub(r'<REP>\w+<WR>\w+</WR></REP><WR>\w+</WR><REP>\w+</REP>', 'REP-B WR-B WR-B REP-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR><WR>\w+</WR><REP>\w+', 'WR-B WR-B REP-B', text.strip()) 
    text = re.sub(r'<WR>\w+</WR><WR>\w+</WR><WR>\w+</WR><REP>\w+<WR>\w+</WR>', 'WR-B WR-B WR-B REP-B WR-B', text.strip())
    text = re.sub(r'<REP>\w+<WR>\w+</WR>\w+', 'REP-B WR-B O', text.strip()) 
    text = re.sub(r'<WR>(.*?)</WR><REP>(.*?)</REP><WR>(.*?)</WR>', 'WR-B REP-B WR-B', text.strip()) 
    text = re.sub(r'<REP>\w+</REP><WR>\w+<WR>\w+</WR></WR><REP>\w+', 'REP-B WR-B WR-B REP-B', text.strip()) 
    text = re.sub(r'<FIL>\w+</FIL><WR>\w+</WR><WR>\w+</WR><WR>\w+</WR>', 'FIL-B WR-B WR-B WR-B', text.strip())
    text = re.sub(r'<FIL>\w+<FIL/><PS/><FIL>\w+</FIL>', 'FIL-B PS-B FIL-B', text.strip())
    text = re.sub(r'<FIL>\w+</FIL><PS/>', 'FIL-B PS-B', text.strip())
    text = re.sub(r'<<FIL>\w+</FIL>', 'FIL-B', text.strip())
    text = re.sub(r'\w+<PS/>', 'PS-B', text.strip())
    text = re.sub(r'<PS/>', 'PS-B', text.strip())
    text = re.sub(r'<PS/>\w+', 'PS-B O', text.strip())
    text = re.sub(r'PS-B\w+', 'PS-B O', text.strip())
    text = re.sub(r'\w+<FIL>', 'FIL-B', text.strip()) 
    text = re.sub(r'<FIL>\w+</FIL>\w+', 'FIL-B O', text.strip())
    text = re.sub(r'<FIL>\s*(.*?)\s*</FIL>', r' <FIL>\1</FIL> ', text.strip())
    text = re.sub(r'<FIL>(.*?)</FIL>', 'FIL-B', text.strip())
    text = re.sub(r'<FIL>(.*?) ', 'FIL-B ', text.strip())
    text = re.sub(r'<FIL>', 'FIL-B', text.strip())
    text = re.sub(r'</FIL>\w+', 'O', text.strip()) 
    text = re.sub(r"<FIL>\w+|FIL\w+|FIL \w+|FIL,", "FIL-B", text.strip()).replace(",", "")
    text = re.sub(r'<REP>\w+(")\w+</REP>', 'REP-B', text.strip()) 
    text = re.sub(r'<REP>\w+</REP>', 'REP-B', text.strip())
    text = re.sub(r'<REP><WR>\w+', 'REP-B WR-B', text.strip()) 
    text = re.sub(r'<REP>\w+-\w+', 'REP-B WR-B', text.strip()) 
    text = re.sub(r'</REP><REP>\w+', 'REP-B', text.strip()) 
    text = re.sub(r'</REP>자', 'REP-B', text.strip()) 
    text = re.sub(r'</REP>창', 'REP-B', text.strip()) 
    text = re.sub(r'<REP>\w+', 'REP-B', text.strip()) 
    text = re.sub(r'<REP>\w+<WR>\w+</WR></REP><WR>\w+</WR>', 'REP-B WR-B WR-B', text.strip()) 
    text = re.sub(r'<REP>(.*?)</REP>', 'REP-B', text.strip())
    text = re.sub(r'<REP>\w+ <PS/>', 'REP-B PS-B', text.strip()) 
    text = re.sub(r'\w+</REP><WR>\w+</WR><WR>\w+</WR><WR>\w+</WR>', 'O WR-B WR-B WR-B', text.strip()) 
    text = re.sub(r'<REP>\w+</REP><REP>\w+<WR>\w+</WR></REP>', 'REP-B REP-B WR-B', text.strip()) 
    text = re.sub(r'<REP>[가-힣?]+', 'REP-B', text.strip())
    text = re.sub(r'<REP> <REP>', 'REP-B', text.strip())
    text = re.sub(r'<REP>[A-Za-z0-9_-가-힣]+', 'REP-B', text.strip()) 
    text = re.sub(r'REP-B\w+', 'REP-B ', text.strip())
    text = re.sub(r'<REP>\s*(.*?)\s*</REP>', r' <REP>\1</REP> ', text.strip())
    text = re.sub(r'<REP>(.*?)</REP>', 'REP-B', text.strip())
    text = re.sub(r"<REP>\w+|REP\w+|REP \w+|REP,", "REP-B", text).replace(",", "")
    text = re.sub(r'\w+<REP>', 'REP-B', text.strip())
    text = re.sub(r'</REP>\w+','O', text.strip()) 
    text = re.sub(r'\w+</REP>','O', text.strip()) 
    text = re.sub(r'</REP>','', text.strip())
    text = re.sub(r'<REP>', 'REP-B', text.strip()) 
    text = re.sub(r'<R','', text.strip())
    text = re.sub(r'<WR>\w+</WR><WR>\w+</WR>', 'WR-B WR-B', text.strip()) 
    text = re.sub(r'</WR></WR>\w+', 'O', text.strip()) 
    text = re.sub(r'</WR>\w+', 'O', text.strip())
    text = re.sub(r'\w+</WR>', 'O', text.strip())
    text = re.sub(r'</WR>', 'O', text.strip())
    text = re.sub(r'<WR>\w+</WR><WR>\w+</WR><WR>\w+</WR>', 'WR-B WR-B WR-B', text.strip())
    text = re.sub(r'<WR>\w+</WR>\w+', 'WR-B O', text.strip())
    text = re.sub(r'<WR>\w+</WR>(.*?)', 'WR-B', text.strip())
    text = re.sub(r'<WR>\w+', 'WR-B', text.strip())
    text = re.sub(r'<WR>\s*(.*?)\s*</WR>', r' <WR>\1</WR> ', text.strip())
    text = re.sub(r'<WR>(.*?)</WR>', 'WR-B', text.strip())
    text = re.sub(r'<WR>(.*?)</WR></REP>', 'WR-B', text.strip()) 
    text = re.sub(r'<WR>(.*?)</WR>(.*?)', 'WR-B', text.strip()) 
    text = re.sub(r'<WR>(.*?) ', 'WR-B ', text.strip())
    text = re.sub(r'</WR>(.*?)</WR>', 'WR-B', text.strip()) 
    text = re.sub(r'</WR>','', text.strip())
    text = re.sub(r'</WR></WR>\w+', 'WR-B', text.strip()) 
    text = re.sub(r'</WR></WR>', '', text.strip()) 
    text = re.sub(r'</WR>\w+</WR>', 'WR-B', text.strip()) 
    text = re.sub(r"<WR>\w+|WR\w+|WR \w+|WR,", "WR-B", text.strip()).replace(",", "")
    text = re.sub(r'</WR>\w+', 'O', text.strip())
    text = re.sub(r'<WR>', 'WR-B', text.strip())
    text = re.sub(r'WR-B</WR', 'WR-B', text.strip())
    text = re.sub(r'WR-BWR-B', 'WR-B WR-B', text.strip()) 
    text = re.sub(r'WR-B O-B', 'WR-B REP-B', text.strip()) 
    text = re.sub(r'<', 'O', text.strip())
    text = re.sub(r'REP-B?', 'REP-B ', text.strip())
    text = re.sub(r'WR-B?', 'WR-B ', text.strip())
    text = re.sub(r"'\w", 'O', text.strip())

    return text

def cleanse_tags(df):
    # 문장 태그 정제
    search_patterns = ['<REP>', '</REP>', '<WR>', '</WR>', '<FIL>', '</FIL>', '?']
    searchtag_patterns = ['?', "'", ',']
    
    for pattern in search_patterns:
        df['Sentence'] = df['Sentence'].str.replace(pattern, '')
        
    for pattern in searchtag_patterns:
        df['Tag'] = df['Tag'].str.replace(pattern, '')

    # 공백이 있는 행 제거
    df = df[df['Sentence'].str.strip() != '']
    
    return df

 # [주의] dataset폴더 안에 라벨링데이터(json) 파일 저장
paths = r'C:\Users\sidhd\Downloads\112.공적말하기 실습 및 평가 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터\라벨링데이터통합\\'
fileNameList = listdir(paths)
len(fileNameList)

main_df = pd.DataFrame()

df= pd.DataFrame(fileNameList, columns=['file_name'])     # 파일 리스트 -> 데이터 프레임화
df=df.astype('string')                                    # df type string 으로 변경

cr = df['file_name'].str.contains('presentation')
data = df[cr]

sub_df = get_data(data, paths)                          # 데이터 가져오는 함수
main_df = pd.concat([main_df, sub_df])            # main_labels에 누적하여 더하기

# 데이터셋 비율 설정
train_ratio = 0.8  # 학습 데이터셋 비율
valid_ratio = 0.1  # 검증 데이터셋 비율

sentences = []

# 문장을 리스트로 추출
for text in main_df['script_tag_txt']:
    sentences.extend(text.split('.'))

# [주의] 랜덤으로 문장 순서 섞도록 설정 가능
# shuffle(sentences)

# 분할 인덱스 계산
train_idx = int(len(sentences) * train_ratio)
valid_idx = int(len(sentences) * (train_ratio + valid_ratio))


# 학습 데이터셋 생성
train_data = []
for sentence in sentences[:train_idx]:
    if sentence.strip():
        train_data.append([sentence.strip(), merge_tags(' '.join(label_text(sentence)))])

# 검증 데이터셋 생성
valid_data = []
for sentence in sentences[train_idx:valid_idx]:
    if sentence.strip():
        valid_data.append([sentence.strip(), merge_tags(' '.join(label_text(sentence)))])

# 테스트 데이터셋 생성
test_data = []
for sentence in sentences[valid_idx:]:
    if sentence.strip():
        test_data.append([sentence.strip(), merge_tags(' '.join(label_text(sentence)))])

# 디렉토리 생성 함수
def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
# 디렉토리 생성
output_dir = "C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/dataset_tagsequence"
create_directory_if_not_exists(output_dir)
# 학습 데이터셋을 CSV 파일로 저장
df_train = pd.DataFrame(train_data, columns=["Sentence", "Tag"])
df_train = cleanse_tags(df_train)
# [주의] 데이터전처리 완료된 학습데이터 파일(csv)을 dataset_tagsequence 폴더에 저장되도록 경로설정
df_train.to_csv("C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/dataset_tagsequence/tag_train_data_17290e.csv", encoding="utf-8-sig", index=False)

# 검증 데이터셋을 CSV 파일로 저장
df_valid = pd.DataFrame(valid_data, columns=["Sentence", "Tag"])
df_valid = cleanse_tags(df_valid)
# [주의] 데이터전처리 완료된 검증데이터 파일(csv)을 dataset_tagsequence 폴더에 저장되도록 경로설정
df_valid.to_csv("C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/dataset_tagsequence/tag_valid_data_2164e.csv", encoding="utf-8-sig", index=False)

#테스트 데이터셋을 CSV 파일로 저장
df_test = pd.DataFrame(test_data, columns=["Sentence", "Tag"])
df_test = cleanse_tags(df_test)
# [주의] 데이터전처리 완료된 테스트데이터 파일(csv)을 dataset_tagsequence 폴더에 저장되도록 경로설정
df_test.to_csv("C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/dataset_tagsequence/tag_test_data_2163e.csv", encoding="utf-8-sig", index=False)


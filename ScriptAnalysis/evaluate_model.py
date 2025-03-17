import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
from sklearn.metrics import classification_report, f1_score
from model import TFBertForTokenClassification  # 모델 정의 파일에서 import
import pandas as pd

# ✅ 1. 모델 불러오기
model_path = "saved_model/kluebert_base_new/variables/variables"
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

# 학습된 태그 정보 로드
label_file = "C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/ner_label_v2.txt"
labels = [label.strip() for label in open(label_file, 'r', encoding='utf-8')]
num_labels = len(labels)
index_to_tag = {idx: tag for idx, tag in enumerate(labels)}

# 모델 정의 후 가중치 불러오기
model = TFBertForTokenClassification(model_name="klue/bert-base", num_labels=num_labels)
model.load_weights(model_path)
print("✅ 모델 가중치 로드 완료!")

# ✅ 2. 테스트 데이터 불러오기
test_ner_df = pd.read_csv("C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/dataset_tagsequence/tag_test_data_2163e.csv")
test_data_sentence = [sent.split() for sent in test_ner_df['Sentence'].values]
test_data_label = [tag.split() for tag in test_ner_df['Tag'].values]

# ✅ 3. 데이터 전처리 함수
def convert_examples_to_features(examples, labels, max_seq_len=128):
    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []
    cls_token, sep_token, pad_token_id = tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token_id

    for example, label in zip(examples, labels):
        tokens, labels_ids = [], []
        for word, label_token in zip(example, label):
            subword_tokens = tokenizer.tokenize(word)
            tokens.extend(subword_tokens)
            labels_ids.extend([index_to_tag.get(label_token, 'O')] + [-100] * (len(subword_tokens) - 1))

        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:max_seq_len - 2]
            labels_ids = labels_ids[:max_seq_len - 2]

        tokens = [cls_token] + tokens + [sep_token]
        labels_ids = [-100] + labels_ids + [-100]

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)

        input_ids.append(input_id + [pad_token_id] * padding_count)
        attention_masks.append(attention_mask + [0] * padding_count)
        token_type_ids.append([0] * max_seq_len)
        data_labels.append(labels_ids + [-100] * padding_count)

    return np.array(input_ids), np.array(attention_masks), np.array(token_type_ids), np.array(data_labels)

# ✅ 4. 테스트 데이터 변환
X_test, attention_masks, token_type_ids, y_test = convert_examples_to_features(test_data_sentence, test_data_label)

# ✅ 5. 예측 수행
y_predicted = model.predict([X_test, attention_masks, token_type_ids])
y_predicted = np.argmax(y_predicted, axis=-1)

# ✅ 6. 평가 (F1-score, classification_report)
def sequences_to_tags(label_ids, pred_ids):
    label_list, pred_list = [], []
    for label_seq, pred_seq in zip(label_ids, pred_ids):
        label_tag, pred_tag = [], []
        for label_index, pred_index in zip(label_seq, pred_seq):
            if label_index != -100:
                label_tag.append(index_to_tag[label_index])
                pred_tag.append(index_to_tag[pred_index])
        label_list.append(label_tag)
        pred_list.append(pred_tag)
    return label_list, pred_list

label_list, pred_list = sequences_to_tags(y_test, y_predicted)
print("\n===== FIL-B 태그의 F1-score =====")
print(f1_score(label_list, pred_list, average=None))
print("\n===== Classification Report =====")
print(classification_report(label_list, pred_list, digits=4))

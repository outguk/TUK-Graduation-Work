# main.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report
from transformers import shape_list, BertTokenizer
import os

# 1. 필수 설정값 초기화
labels = [label.strip() for label in open('C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/ner_label_v2.txt', 'r', encoding='utf-8')]
tag_to_index = {tag: index for index, tag in enumerate(labels)}
index_to_tag = {index: tag for index, tag in enumerate(labels)}
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

# 2. 저장된 모델 로드
loaded_model = tf.keras.models.load_model('saved_model/kluebert_base_new')

# 3. 전처리 함수 정의
def convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer,
                                               pad_token_id_for_segment=0,
                                               pad_token_id_for_label=-100):
    # (기존 코드와 동일)
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    input_ids, attention_masks, token_type_ids, label_masks = [], [], [], []

    for example in tqdm(examples):
        tokens = []
        label_mask = []
        for one_word in example:
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            label_mask.extend([0] + [pad_token_id_for_label] * (len(subword_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            label_mask = label_mask[:(max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        label_mask += [pad_token_id_for_label]
        tokens = [cls_token] + tokens
        label_mask = [pad_token_id_for_label] + label_mask

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label_mask = label_mask + ([pad_token_id_for_label] * padding_count)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        label_masks.append(label_mask)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    label_masks = np.asarray(label_masks, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), label_masks

# 4. 예측 함수 정의 (모델 객체 교체)
def ner_prediction(examples, max_seq_len, tokenizer):
    examples = [sent.split() for sent in examples]
    X_pred, label_masks = convert_examples_to_features_for_prediction(
        examples, max_seq_len=128, tokenizer=tokenizer
    )
    y_predicted = loaded_model.predict(X_pred)  # loaded_model 사용
    y_predicted = np.argmax(y_predicted, axis=2)

    pred_list = []
    result_list = []

    for i in range(len(label_masks)):
        pred_tag = []
        for label_index, pred_index in zip(label_masks[i], y_predicted[i]):
            if label_index != -100:
                pred_tag.append(index_to_tag[pred_index])
        pred_list.append(pred_tag)

    for example, pred in zip(examples, pred_list):
        one_sample_result = []
        for one_word, label_token in zip(example, pred):
            one_sample_result.append((one_word, label_token))
        result_list.append(one_sample_result)

    return result_list

# 5. 예측 실행
sent1 = '그날 제가 그 약속을 어겼을 때 저의 무관심한 태도와 사과 없이 그저 무시한 무시한 음 어 무시한 것이 얼마나 상대방에게 어 실망과 불편함을 안겼는지 깨달았습니다'
sent2 = '어  성범죄자나 폭력범죄자의 신상공개는 사회적 위험성을  최소화하기 최소화하기  위한 중요한  정책이고   사회  어  사회  안전과 보호를  강강화하는데  중요한 역할을 할 수 있습니다'

results = ner_prediction([sent1, sent2], max_seq_len=128, tokenizer=tokenizer)
for idx, result in enumerate(results):
    print(f"Sentence {idx+1} 결과:")
    for word, tag in result:
        print(f"{word}: {tag}")
    print("\n" + "="*50 + "\n")
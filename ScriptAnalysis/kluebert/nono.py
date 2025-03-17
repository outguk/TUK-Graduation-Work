# main.py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel

# 1. 필수 설정값 초기화
labels = [label.strip() for label in open('C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/ner_label_v2.txt', 'r', encoding='utf-8')]
index_to_tag = {index: tag for index, tag in enumerate(labels)}
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

# 2. 커스텀 모델 클래스 재정의 (학습 코드와 동일)
class TFBertForTokenClassification(tf.keras.Model):
    def __init__(self, model_name, num_labels, dropout_rate=0.1):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(
            num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
            name='classifier'
        )

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

# 3. 모델 생성 및 가중치 로드
model = TFBertForTokenClassification("klue/bert-base", num_labels=len(labels))
model.load_weights('saved_model/kluebert_base_new/variables/variables').expect_partial()

# 4. 전처리 함수
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

# 5. 예측 함수 (model → model 사용)
def ner_prediction(examples, max_seq_len, tokenizer):
    examples = [sent.split() for sent in examples]
    X_pred, label_masks = convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer)
    y_predicted = model.predict(X_pred)
    y_predicted = np.argmax(y_predicted, axis=2)

    pred_list = []
    for i in range(len(label_masks)):
        pred_tag = []
        for label_idx, pred_idx in zip(label_masks[i], y_predicted[i]):
            if label_idx != -100:
                pred_tag.append(index_to_tag[pred_idx])
        pred_list.append(pred_tag)

    result_list = []
    for example, pred in zip(examples, pred_list):
        result_list.append(list(zip(example, pred)))
    return result_list

# 6. 예측 실행
sent1 = '그날 제가 그 약속을 어겼을 때 저의 무관심한 태도와 사과 없이 그저 무시한 무시한 음 어 무시한 것이 얼마나 상대방에게 어 실밍과 불편함을 안겼는지 깨달았습니다'
sent2 = '어  성범죄자나 폭력범죄자의 신상공개는 사회적 위험성을  최소화하기 최소화하기  위한 중요한  정책이고   사회  어  사회  안전과 보호를  깅화하는데  중요한 역할을 할 수 있습니다'

results = ner_prediction([sent1, sent2], max_seq_len=128, tokenizer=tokenizer)

for idx, result in enumerate(results):
    print(f"Sentence {idx+1} 결과:")
    for word, tag in result:
        print(f"{word}: {tag}")
    print("\n" + "="*50 + "\n")
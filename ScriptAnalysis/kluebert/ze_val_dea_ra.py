# main.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import shape_list, BertTokenizer, TFBertModel
from tensorflow.keras.callbacks import EarlyStopping
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# GPU, CPU 장치 확인
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

# 학습에 사용될 GPU 장치 확인
print('GPU Device: ', tf.config.experimental.list_physical_devices('GPU'))


# 데이터 로드 및 전처리
# [주의] 데이터전처리 완료된 학습데이터,검증데이터,테스트데이터 파일(csv)을 dataset_tagsequence 폴더에서 로드하기 위한 경로 설정 필요
# [주의] 데이터전처리 완료된 학습데이터,검증데이터,테스트데이터 파일(csv)을 dataset_tagsequence 폴더에서 로드하기 위한 경로 설정 필요
train_ner_df = pd.read_csv("C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/dataset_tagsequence/tag_train_data_17290e.csv")
valid_ner_df = pd.read_csv("C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/dataset_tagsequence/tag_valid_data_2164e.csv")
test_ner_df = pd.read_csv("C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/dataset_tagsequence/tag_test_data_2163e.csv")

train_data_sentence = [sent.split() for sent in train_ner_df['Sentence'].values]
test_data_sentence = [sent.split() for sent in test_ner_df['Sentence'].values]
valid_data_sentence = [sent.split() for sent in valid_ner_df['Sentence'].values]
train_data_label = [tag.split() for tag in train_ner_df['Tag'].values]
valid_data_label = [tag.split() for tag in valid_ner_df['Tag'].values]
test_data_label = [tag.split() for tag in test_ner_df['Tag'].values]

# 개체명 태깅 정보 : ['O', 'FIL-B', 'REP-B', 'PS-B', 'WR-B']
# [주의] 태그가 정의된 파일(ner_label_v2.txt)을 dataset_nerlabel 폴더에서 로드하기 위한 경로 설정 필요
labels = [label.strip() for label in open('C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/ner_label_v2.txt', 'r', encoding='utf-8')]

tag_to_index = {tag: index for index, tag in enumerate(labels)}
index_to_tag = {index: tag for index, tag in enumerate(labels)}

# 개체명 태깅 정보의 개수: 5 (tag_size)
tag_size = len(tag_to_index)

# 토크나이저를 통한 형태소 분리 - KLUE-BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

def convert_examples_to_features(examples, labels, max_seq_len, tokenizer,
                                                  pad_token_id_for_segment=0,
                                                  pad_token_id_for_label=-100):
  cls_token = tokenizer.cls_token
  sep_token = tokenizer.sep_token
  pad_token_id = tokenizer.pad_token_id
  input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []
  for example, label in tqdm(zip(examples, labels), total=len(examples)):
    tokens = []
    labels_ids = []
    for one_word, label_token in zip(example, label):
      # 하나의 단어에 대해서 서브워드로 토큰화
      subword_tokens = tokenizer.tokenize(one_word)
      tokens.extend(subword_tokens)
      # 서브워드 중 첫번째 서브워드만 개체명 레이블을 부여하고 그 외에는 -100으로 채운다.
      labels_ids.extend([tag_to_index[label_token]]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

    # [CLS]와 [SEP]를 후에 추가할 것을 고려하여 최대길이를 초과하는 샘플의 경우 max_seq_len - 2의 길이로 변환.
    # ex) max_seq_len = 64라면 길이가 62보다 긴 샘플은 뒷부분을 자르고 길이 62로 변환.
    special_tokens_count = 2
    if len(tokens) > max_seq_len - special_tokens_count:
      tokens = tokens[:(max_seq_len - special_tokens_count)]
      labels_ids = labels_ids[:(max_seq_len - special_tokens_count)]
    # [SEP]를 추가하는 코드
    # 1. 토큰화 결과의 맨 뒷부분에 [SEP]토큰 추가
    # 2. 레이블에도 맨 뒷부분에 -100 추가.
    tokens += [sep_token]
    labels_ids += [pad_token_id_for_label]

    # [CLS]를 추가하는 코드
    # 1. 토큰화 결과의 앞부분에 [CLS] 토큰 추가
    # 2. 레이블의 맨 앞부분에도 -100 추가.
    tokens = [cls_token] + tokens
    labels_ids = [pad_token_id_for_label] + labels_ids

    # 정수인코딩
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    # 어텐션마스크 생성
    attention_mask = [1] * len(input_id)
    # 정수인코딩에 추가할 패딩길이 연산
    padding_count = max_seq_len - len(input_id)
    # 정수인코딩 ,어텐션 마스크에 패딩 추가
    input_id = input_id + ([pad_token_id] * padding_count)
    attention_mask = attention_mask + ([0] * padding_count)
    # 세그먼트 인코딩.
    token_type_id = [pad_token_id_for_segment] * max_seq_len
    # 레이블 패딩. (단, 이 경우는 패딩 토큰의 ID가 -100)
    label = labels_ids + ([pad_token_id_for_label] * padding_count)

    assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
    assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
    assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
    assert len(label) == max_seq_len, "Error with labels length {} vs {}".format(len(label), max_seq_len)

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)
    data_labels.append(label)

  input_ids = np.array(input_ids, dtype=int)
  attention_masks = np.array(attention_masks, dtype=int)
  token_type_ids = np.array(token_type_ids, dtype=int)
  data_labels = np.asarray(data_labels, dtype=np.int32)

  return (input_ids, attention_masks, token_type_ids), data_labels


X_train, y_train = convert_examples_to_features(train_data_sentence,
train_data_label, max_seq_len=128, tokenizer=tokenizer)
X_valid, y_valid = convert_examples_to_features(valid_data_sentence, valid_data_label,
max_seq_len=128, tokenizer=tokenizer)
X_test, y_test = convert_examples_to_features(test_data_sentence, test_data_label,
max_seq_len=128, tokenizer=tokenizer)


class TFBertForTokenClassification(tf.keras.Model):
  def __init__(self, model_name, num_labels, dropout_rate=0.1): # 드롭아웃 설정 (기본값: 0.1)
    super(TFBertForTokenClassification, self).__init__()
    self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.classifier = tf.keras.layers.Dense(num_labels,
                                            kernel_initializer=tf.keras.
                                            initializers.TruncatedNormal
                                            (0.02),
                                            name='classifier')

  def call(self, inputs):
    input_ids, attention_mask, token_type_ids = inputs
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
      token_type_ids=token_type_ids)

    # 전체 시퀀스에 대해서 분류해야하므로 outputs[0]임에 주의
    all_output = outputs[0]
    all_output = self.dropout(all_output)  # dropout 레이어 추가
    prediction = self.classifier(all_output)

    return prediction
  
labels = tf.constant([[-100, 2, 1, -100]])
logits = tf.constant([[[0.8, 0.1, 0.1], [0.06, 0.04, 0.9], [0.75, 0.1, 0.15],
  [0.4, 0.5, 0.1]]])

active_loss = tf.reshape(labels, (-1,)) != -100

reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])),
  active_loss)

labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

def compute_loss(labels, logits):
  # 다중 클래스 분류 문제에서 소프트맥스 함수 미사용시 from_logits=True로 설정.
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  # -100의 값을 가진 정수에 대해서는 오차를 반영하지 않도록 labels를 수정.
  active_loss = tf.reshape(labels, (-1,)) != -100

  # activa_loss로부터 reduced_logits과 labels를 각각 얻는다.
  reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2]))
    , active_loss)
  labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
  return loss_fn(labels, reduced_logits)

# 6단계: 모델 학습 및 평가 - KLUE-BERT Model

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

# 모델 초기화 및 학습
with strategy.scope():
  model = TFBertForTokenClassification("klue/bert-base", num_labels=tag_size) 
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  model.compile(optimizer=optimizer, loss=compute_loss)

class F1score(tf.keras.callbacks.Callback):
  def __init__(self, X_test, y_test):
    self.X_test = X_test
    self.y_test = y_test

  def sequences_to_tags(self, label_ids, pred_ids):
    label_list = []
    pred_list = []

    for i in range(0, len(label_ids)):
      label_tag = []
      pred_tag = []

      # 레이블의 값이 -100인 경우는 F1 score 계산시에도 제외
      # ex) 레이블디코딩과정
      # label_index : [1 -100 2 -100] ===> [1 2] ===> label_tag : [PER-B PER-I]
      for label_index, pred_index in zip(label_ids[i], pred_ids[i]):
        if label_index != -100:
          label_tag.append(index_to_tag[label_index])
          pred_tag.append(index_to_tag[pred_index])

      label_list.append(label_tag)
      pred_list.append(pred_tag)

    return label_list, pred_list

  # 에포크가 끝날때마다 실행되는 함수}
  def on_epoch_end(self, epoch, logs={}):
    y_predicted = self.model.predict(self.X_test)
    y_predicted = np.argmax(y_predicted, axis = 2)

    label_list, pred_list = self.sequences_to_tags(self.y_test, y_predicted)

    score = f1_score(label_list, pred_list, suffix=True)
    print(' - f1: {:04.2f}'.format(score * 100))
    print(classification_report(label_list, pred_list, suffix=True))


# 7단계: 테스트데이터셋 평가(평가지표: F1-Score)

# 모델 평가
# 모델학습 과정에서의 테스트데이터셋 대상 모델 성능 모니터링
f1_score_report = F1score(X_test, y_test)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# 조기 종료
early_stopping = EarlyStopping(
    monitor='val_loss', 
    #patience=5,
    patience=10,

    restore_best_weights=True
)

# 모델 체크포인트
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 모델 가중치를 저장하는 콜백 
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# 학습
model.fit(
  #X_train, y_train, epochs=10, batch_size=32,
  X_train, y_train, epochs=20, batch_size=16,

  validation_data=(X_valid, y_valid),
  callbacks = [f1_score_report, early_stopping, cp_callback] 
)

# 모델 아키텍처 확인
model.summary()

# 모델 저장
model.save('saved_model/kluebert_base_new')

def convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer,
                                                  pad_token_id_for_segment=0,
                                                  pad_token_id_for_label=-100):
  cls_token = tokenizer.cls_token
  sep_token = tokenizer.sep_token
  pad_token_id = tokenizer.pad_token_id

  input_ids, attention_masks, token_type_ids, label_masks = [], [], [], []

  for example in tqdm(examples):
    tokens = []
    label_mask = []
    for one_word in example:
      # 하나의 단어에 대해서 서브워드로 토큰화
      subword_tokens = tokenizer.tokenize(one_word)
      tokens.extend(subword_tokens)
      # 서브워드 중 첫번째 서브워드만 개체명 레이블을 부여하고 그 외에는 -100으로 채운다.
      label_mask.extend([0]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

    # [CLS]와 [SEP]를 후에 추가할 것을 고려하여 최대길이를 초과하는 샘플의 경우 max_seq_len - 2의 길이로 변환.
    # ex) max_seq_len = 64라면 길이가 62보다 긴 샘플은 뒷 부분을 자르고 길이 62로 변 환.
    special_tokens_count = 2
    if len(tokens) > max_seq_len - special_tokens_count:
      tokens = tokens[:(max_seq_len - special_tokens_count)]
      label_mask = label_mask[:(max_seq_len - special_tokens_count)]

    # [SEP]를 추가하는 코드
    # 1. 토큰화 결과의 맨 뒷부분에 [SEP] 토큰 추가
    # 2. 레이블에도 맨뒷부분에 -100 추가.
    tokens += [sep_token]
    label_mask += [pad_token_id_for_label]

    # [CLS]를 추가하는 코드
    # 1. 토큰화 결과의 앞부분에 [CLS] 토큰 추가
    # 2. 레이블의 맨앞부분에도 -100 추가.
    tokens = [cls_token] + tokens
    label_mask = [pad_token_id_for_label] + label_mask

    # 정수인코딩
    input_id = tokenizer.convert_tokens_to_ids(tokens)

    # 어텐션 마스크 생성
    attention_mask = [1] * len(input_id)

    # 정수 인코딩에 추가할 패딩 길이 연산
    padding_count = max_seq_len - len(input_id)

    # 정수인코딩 ,어텐션 마스크에 패딩 추가
    input_id = input_id + ([pad_token_id] * padding_count)
    attention_mask = attention_mask + ([0] * padding_count)

    # 세그먼트 인코딩.
    token_type_id = [pad_token_id_for_segment] * max_seq_len

    # 레이블 패딩. (단, 이 경우는 패딩 토큰의 ID가 -100)
    label_mask = label_mask + ([pad_token_id_for_label] * padding_count)

    assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
    assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
    assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
    assert len(label_mask) == max_seq_len, "Error with labels length {} vs {}".format(len(label_mask), max_seq_len)

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)
    label_masks.append(label_mask)

  input_ids = np.array(input_ids, dtype=int)
  attention_masks = np.array(attention_masks, dtype=int)
  token_type_ids = np.array(token_type_ids, dtype=int)
  label_masks = np.asarray(label_masks, dtype=np.int32)

  return (input_ids, attention_masks, token_type_ids), label_masks


X_pred, label_masks = convert_examples_to_features_for_prediction(test_data_sentence[:5], max_seq_len=128, tokenizer=tokenizer)


def ner_prediction(examples, max_seq_len, tokenizer):
  examples = [sent.split() for sent in examples]
  X_pred, label_masks = convert_examples_to_features_for_prediction(examples,
  max_seq_len=128, tokenizer=tokenizer)
  y_predicted = model.predict(X_pred)
  y_predicted = np.argmax(y_predicted, axis = 2)

  pred_list = []
  result_list = []

  for i in range(0, len(label_masks)):
    pred_tag = []
    # ex) 모델의 예측값 디코딩 과정
    # 예측값(y_predicted)에서 레이블마스크(label_masks)의 값이 -100인 동일 위치의 값을 삭제
    # label_masks : [-100 0 -100 0 -100]
    # y_predicted : [ 0 1 0 2 0 ] ==> [1 2] ==> 최 종 예 측(pred_tag) : [PER-B PER-I]
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


# 8단계: 테스트데이터셋 중 샘플 문장 2개 평가 및 검증

sent1 = '그날 제가 그 약속을 어겼을 때 저의 무관심한 태도와 사과 없이 그저 무시한 무시한 음 어 무시한 것이 얼마나 상대방에게 어 실망과 불편함을 안겼는지 깨달았습니다'
sent2 = '어  성범죄자나 폭력범죄자의 신상공개는 사회적 위험성을  최소화하기 최소화하기  위한 중요한  정책이고   사회  어  사회  안전과 보호를  강강화하는데  중요한 역할을 할 수 있습니다'
test_samples = [sent1, sent2]
result_list = ner_prediction(test_samples, max_seq_len=128, tokenizer=tokenizer)
print(result_list)

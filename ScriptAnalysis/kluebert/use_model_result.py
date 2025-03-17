import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm

print("GPU 사용 가능:", tf.config.list_physical_devices('GPU'))

class TFBertForTokenClassification(tf.keras.Model):
    def __init__(self, model_name="klue/bert-base", num_labels=6, dropout_rate=0.1):
        super(TFBertForTokenClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(
            num_labels, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), name='classifier'
        )

    def call(self, inputs, training=False):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        all_output = self.dropout(outputs[0], training=training)
        return self.classifier(all_output)

labels = ['O', 'FIL-B', 'REP-B', 'PS-B', 'WR-B', 'UNK']
index_to_tag = {idx: tag for idx, tag in enumerate(labels)}
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

model = TFBertForTokenClassification(model_name="klue/bert-base", num_labels=6)
model.load_weights('saved_model/kluebert_base_new/variables/variables')

def split_long_text(text, max_seq_len=128, overlap=20):
    words = text.split()
    segments = []
    start = 0
    while start < len(words):
        end = min(start + max_seq_len - 2, len(words))
        segments.append(words[start:end])
        if end == len(words):
            break
        start = end - overlap
    return segments

def convert_examples_to_features(examples, max_seq_len, tokenizer):
    cls_token, sep_token, pad_token_id = tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token_id
    input_ids, attention_masks, token_type_ids = [], [], []
    
    for example in tqdm(examples):
        tokens = []
        for word in example:
            tokens.extend(tokenizer.tokenize(word))
        
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:max_seq_len - 2]
        
        tokens = [cls_token] + tokens + [sep_token]
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        padding_count = max_seq_len - len(input_id)
        
        input_ids.append(input_id + [pad_token_id] * padding_count)
        attention_masks.append([1] * len(input_id) + [0] * padding_count)
        token_type_ids.append([0] * max_seq_len)
    
    return np.array(input_ids, dtype=int), np.array(attention_masks, dtype=int), np.array(token_type_ids, dtype=int)

def predict_ner(texts, max_seq_len=128, overlap=20):
    all_results = []
    for text in texts:
        segments = split_long_text(text, max_seq_len, overlap)
        (input_ids, attention_masks, token_type_ids) = convert_examples_to_features(segments, max_seq_len, tokenizer)
        logits = model([input_ids, attention_masks, token_type_ids], training=False)
        predictions = np.argmax(logits, axis=-1)
        
        result = []
        for words, pred in zip(segments, predictions):
            tags = [index_to_tag.get(idx, 'UNK') for idx in pred[:len(words)]]
            result.extend(list(zip(words, tags)))
        all_results.append(result)
    return all_results

if __name__ == "__main__":
    file_path = "C:\\Users\\sidhd\\Tukgu\\TUK-Graduation-Work\\output_transcription.txt"
    output_file = "C:\\Users\\sidhd\\Tukgu\\TUK-Graduation-Work\\ner_results.txt"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        test_samples = [line.strip() for line in file.readlines() if line.strip()]
    
    predictions = predict_ner(test_samples)
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for i, sentence_result in enumerate(predictions):
            out_file.write(f"\n문장 {i+1} 결과:\n")
            print(f"\n문장 {i+1} 결과:")
            for word, tag in sentence_result:
                result_line = f"{word:15} => {tag}"
                print(result_line)
                out_file.write(result_line + '\n')
    
    print(f"\n결과가 {output_file} 파일에 저장되었습니다.")

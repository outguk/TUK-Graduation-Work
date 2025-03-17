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

def compute_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    active_loss = tf.reshape(labels, (-1,)) != -100
    reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, tf.shape(logits)[2])), active_loss)
    labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
    return loss_fn(labels, reduced_logits)

labels = [label.strip() for label in open('C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ScriptAnalysis/kluebert/ner_label_v2.txt', 'r', encoding='utf-8')]
index_to_tag = {index: tag for index, tag in enumerate(labels)}
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

model = tf.keras.models.load_model(
    'C:/Users/sidhd/Tukgu/TUK-Graduation-Work/saved_model/kluebert_base_new',
    custom_objects={
        "TFBertForTokenClassification": TFBertForTokenClassification,
        "compute_loss": compute_loss
    }
)

def convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    
    input_ids, attention_masks, token_type_ids = [], [], []
    
    for example in tqdm(examples):
        tokens = [cls_token] + tokenizer.tokenize(example)[: max_seq_len - 2] + [sep_token]
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        token_type_id = [0] * len(input_id)

        padding_count = max_seq_len - len(input_id)
        input_id += [pad_token_id] * padding_count
        attention_mask += [0] * padding_count
        token_type_id += [0] * padding_count

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
    
    return (np.array(input_ids), np.array(attention_masks), np.array(token_type_ids))

def ner_prediction(examples, max_seq_len=128, tokenizer=tokenizer):
    X_pred = convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer)
    y_predicted = model.predict(X_pred)
    y_predicted = np.argmax(y_predicted, axis=2)

    result_list = []
    for example, pred in zip(examples, y_predicted):
        words = example.split()
        pred_tags = pred[:len(words)]  # Ensure length matches
        result_list.append([(word, index_to_tag[pred_idx]) for word, pred_idx in zip(words, pred_tags)])
    return result_list

if __name__ == "__main__":
    file_path = "C:/Users/sidhd/Tukgu/TUK-Graduation-Work/output_transcription.txt"
    output_file = "C:/Users/sidhd/Tukgu/TUK-Graduation-Work/ner_results.txt"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        test_samples = [line.strip() for line in file.readlines() if line.strip()]
    
    predictions = ner_prediction(test_samples)
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for i, sentence_result in enumerate(predictions):
            out_file.write(f"\n문장 {i+1} 결과:\n")
            print(f"\n문장 {i+1} 결과:")
            for word, tag in sentence_result:
                result_line = f"{word:15} => {tag}"
                print(result_line)
                out_file.write(result_line + '\n')
    
    print(f"\n결과가 {output_file} 파일에 저장되었습니다.")
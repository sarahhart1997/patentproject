from flask import Flask, render_template, request
from transformers import BertTokenizer, TFBertForSequenceClassification
from datasets import load_dataset
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def patentability_score(abstract, claims):
    # Tokenize and encode the data
    max_length = 128  # Set an appropriate maximum sequence length
    encoded_abstract = tokenizer(abstract, padding=True, truncation=True, max_length=max_length, return_tensors='tf')
    encoded_claims = tokenizer(claims, padding=True, truncation=True, max_length=max_length, return_tensors='tf')

    input_ids = [encoded_abstract.input_ids[0], encoded_claims.input_ids[0]]
    attention_mask = [encoded_abstract.attention_mask[0], encoded_claims.attention_mask[0]]

    input_ids = tf.convert_to_tensor(input_ids)
    attention_mask = tf.convert_to_tensor(attention_mask)

    # Make predictions using the model
    predictions = model.predict([input_ids, attention_mask])
    patentability_score = predictions.logits[0][1].numpy()  # Assuming '1' corresponds to the positive class

    return patentability_score

# Load the dataset
dataset_dict = load_dataset('HUPD/hupd',
    name='sample',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather",
    icpr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-21',
    val_filing_start_date='2016-01-22',
    val_filing_end_date='2016-01-31',
)

# Retrieve based on application number
def get_patent_sections(application_number):
    abstract = "Abstract text"  
    claims = "Claims text"  
    return abstract, claims

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        application_number = request.form['application_number']

        # Retrieve patent sections (abstract and claims) based on the selected application number
        abstract, claims = get_patent_sections(application_number)

        # Calculate the patentability score
        score = patentability_score(abstract, claims)

        return render_template('index.html', abstract=abstract, claims=claims, score=score)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

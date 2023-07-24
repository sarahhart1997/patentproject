import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@st.cache(allow_output_mutation=True)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    with st.spinner("Thinking..."):
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=1).item()
        sentiment = "Positive" if predictions == 1 else "Negative"
        return sentiment


def main():
    st.title("Sentiment Analysis App")
    st.write("Enter text and select a model")

    text = st.text_area("Your text here")
    model_name = st.selectbox(
        "Select a pretrained model:",
        [
            "bert-base-uncased",
            "bert-base-cased",
            "distilbert-base-uncased",
            "distilbert-base-cased",
            "roberta-base",
            "roberta-large",
        ],
    )

    tokenizer, model = load_model(model_name)

    if st.button("Analyze"):
        if text.strip() != "":
            sentiment = analyze_sentiment(text, tokenizer, model)
            confidence = result['score']
            confidence_percentage = round(confidence * 100, 2)
            st.success(f"Sentiment: {sentiment}")
            st.success(f"Confidence: {confidence_percentage}%")
        else:
            st.warning("You must enter text.")


if __name__ == "__main__":
    main()

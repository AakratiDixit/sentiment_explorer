# streamlit_app.py

import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 


# 1. Load HuggingFace sentiment‑analysis pipeline
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return lambda text: classify_sentiment(text, model, tokenizer)

def classify_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    labels = ["Negative", "Neutral", "Positive"]
    scores = {labels[i]: float(probs[i]) for i in range(3)}
    top_label = max(scores, key=scores.get)
    return {"label": top_label, "score": scores[top_label], "scores": scores}


sentiment = load_sentiment_model()

# 2. App layout
st.title("🎀 Sentiment Explorer")
st.write("Paste any text below to see its sentiment and visualize the words.")

user_text = st.text_area("Your text here", height=200)

# 3. When user submits...
if st.button("Analyze Sentiment") and user_text.strip():
    with st.spinner("Analyzing…"):
        result = sentiment(user_text)
        label = result["label"]
        score = result["score"]

    # 4. Display prediction
    st.subheader("Prediction")
    st.markdown(f"**{label}** (confidence: {score:.4f})")

    # 5. Bar chart of probability
    # 5. Bar chart of probability
    df = pd.DataFrame(result["scores"].items(), columns=["label", "score"]).set_index("label")

    st.subheader("Sentiment Probabilities")
    st.bar_chart(df)

    # 6. Word‐cloud of the input
    st.subheader("Word Cloud")
    wc = WordCloud(width=800, height=400, background_color="white") \
             .generate(user_text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # 7. Footer
    st.markdown("---")
    st.write("Built with ❤️ using Streamlit & 🌟 HuggingFace transformers")

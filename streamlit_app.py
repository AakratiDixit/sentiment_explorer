# streamlit_app.py

import streamlit as st
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load HuggingFace sentiment‑analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment = load_sentiment_model()

# 2. App layout
st.title("🎀 Sentiment Explorer")
st.write("Paste any text below to see its sentiment and visualize the words.")

user_text = st.text_area("Your text here", height=200)

# 3. When user submits...
if st.button("Analyze Sentiment") and user_text.strip():
    with st.spinner("Analyzing…"):
        result = sentiment(user_text)[0]
        label = result["label"]
        score = result["score"]

    # 4. Display prediction
    st.subheader("Prediction")
    st.markdown(f"**{label}** (confidence: {score:.4f})")

    # 5. Bar chart of probability
    df = pd.DataFrame([{ "label": label, "score": score }]).set_index("label")
    st.subheader("Confidence Score")
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

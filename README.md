# 🎀 Sentiment Explorer

**Sentiment Explorer** is a simple web app built with [Streamlit](https://streamlit.io/) that analyzes the **sentiment** of any input text using Hugging Face's `transformers` and visualizes it with a **word cloud** and **confidence chart**.

<div align="center">
  <img src="https://img.shields.io/badge/Made%20with-Streamlit-orange" />
  <img src="https://img.shields.io/badge/Model-HuggingFace-blue" />
  <img src="https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20WordCloud-green" />
</div>

---

## ✨ Features

- 📥 Accepts user-inputted text
- 🤖 Predicts sentiment (Positive / Negative / Neutral)
- 📊 Displays model confidence as a bar chart
- ☁️ Generates a word cloud from the input
- ⚡ Fast and interactive UI powered by Streamlit

---

## 🔧 How It Works

- Uses HuggingFace's `pipeline("sentiment-analysis")` for NLP
- Renders visual elements using `matplotlib` and `wordcloud`
- Caches the model to optimize performance using `st.cache_resource`

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/sentiment-explorer.git
cd sentiment-explorer


###2. Install dependencies
pip install -r requirements.txt


###3. Run the app
streamlit run streamlit_app.py
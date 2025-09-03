# WhatsApp Chat Analyzer 📊💬

A Streamlit app to analyze exported WhatsApp chat data with insights like:
- Total messages, words, media, and links
- Most active users
- WordCloud & common words
- Emoji analysis
- Activity heatmaps

## 🚀 How to run locally
```bash
# clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# (optional) create venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run streamlit app
streamlit run streamlit_app.py

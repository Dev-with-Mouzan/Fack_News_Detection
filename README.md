# 🗞️ TruthSeeker AI: Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-009688?style=flat&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0+-EE4C2C?style=flat&logo=xgboost&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-83B81A?style=flat)

TruthSeeker AI is a professional fake news detection system built using state-of-the-art machine learning techniques and a premium web interface. It leverages an **XGBoost Classifier** to analyze news articles and determine their credibility with high accuracy.

---

## ✨ Key Features

- **Advanced ML Model**: Powered by an XGBoost Classifier trained on thousands of real and fake news articles.
- **Premium User Interface**: Modern, glassmorphism-inspired UI with real-time feedback and smooth animations.
- **Fast Prediction Engine**: Built on FastAPI for high-performance inference.
- **Comprehensive Preprocessing**: custom NLP pipeline using NLTK for text cleaning, tokenization, and stop-word removal.

---

## 🛠️ Technology Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/)
- **Natural Language Processing**: [NLTK](https://www.nltk.org/)
- **Data Analysis**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Serialization**: [Joblib](https://joblib.readthedocs.io/)
- **Frontend**: Vanilla HTML5, CSS3, and JavaScript (with [Inter](https://fonts.google.com/specimen/Inter) typography)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- `pip` package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dev-with-Mouzan/Fack_News_Detection.git
   cd Fack_News_Detection
   ```

2. **Set up a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**:
   The application will automatically attempt to download required NLTK data on first run, but you can also do it manually:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('punkt_tab')
   ```

### 🏃 Running the Application

Start the FastAPI server using Uvicorn:

```bash
python app.py
```
Or use the direct Uvicorn command:
```bash
uvicorn app:app --reload
```

Navigate to `http://127.0.0.1:8000` in your browser to access the TruthSeeker AI dashboard.

---

## 📁 Project Structure

```text
Final_year/
├── app.py                   # FastAPI main application & UI
├── final_model(XGBoost).pkl # Trained XGBoost classification model
├── vectorizer.pkl           # TF-IDF Vectorizer for text transformation
├── model.ipynb              # Training notebook & experiments
├── requirements.txt         # Project dependencies
├── DataSet/                 # Training dataset (True.csv & Fake.csv)
└── README.md                # Project documentation
```

> [!IMPORTANT]
> Ensure both `final_model(XGBoost).pkl` and `vectorizer.pkl` are present in the root directory before running `app.py`.

---

## 🧪 Model Performance

The XGBoost model achieved excellent results during training:
- **Accuracy**: ~99.7%
- **F1-Score**: 1.00 (on test data)

You can explore the full training process and evaluation metrics in the `model.ipynb` notebook.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Made with ❤️ by [Mouzan Raza](https://github.com/Dev-with-Mouzan)

import re
import string
import joblib
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load model and vectorizer
try:
    model = joblib.load("final_model(XGBoost).pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")

app = FastAPI()

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def preprocess(text):
    text = clean_text(text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_words])
    return text

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fake News Detector | Premium Analysis</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --bg: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --success: #22c55e;
                --error: #ef4444;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: 'Inter', sans-serif;
            }

            body {
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 2rem;
                overflow-x: hidden;
            }

            .container {
                max-width: 800px;
                width: 100%;
                z-index: 10;
            }

            header {
                text-align: center;
                margin-bottom: 3rem;
            }

            h1 {
                font-size: 3.5rem;
                font-weight: 800;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
                letter-spacing: -0.05em;
            }

            p.subtitle {
                color: var(--text-muted);
                font-size: 1.1rem;
            }

            .card {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 2.5rem;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }

            textarea {
                width: 100%;
                height: 200px;
                background: rgba(15, 23, 42, 0.5);
                border: 2px solid rgba(255, 255, 255, 0.05);
                border-radius: 16px;
                padding: 1.5rem;
                color: var(--text);
                font-size: 1rem;
                resize: none;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                margin-bottom: 1.5rem;
            }

            textarea:focus {
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2);
                background: rgba(15, 23, 42, 0.8);
            }

            button {
                width: 100%;
                background: var(--primary);
                color: white;
                border: none;
                padding: 1.25rem;
                border-radius: 16px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.75rem;
            }

            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
                filter: brightness(1.1);
            }

            button:active {
                transform: translateY(0);
            }

            #result-container {
                margin-top: 2rem;
                padding: 1.5rem;
                border-radius: 16px;
                display: none;
                animation: slideUp 0.5s ease-out;
            }

            .result-text {
                font-size: 1.5rem;
                font-weight: 700;
                text-align: center;
            }

            .result-real {
                background: rgba(34, 197, 94, 0.1);
                border: 1px solid var(--success);
                color: var(--success);
            }

            .result-fake {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid var(--error);
                color: var(--error);
            }

            @keyframes slideUp {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }

            /* Loading spinner */
            .loader {
                width: 24px;
                height: 24px;
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 1s linear infinite;
                display: none;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .blur-bg {
                position: fixed;
                width: 500px;
                height: 500px;
                background: var(--primary);
                filter: blur(150px);
                border-radius: 50%;
                opacity: 0.15;
                z-index: -1;
                top: -100px;
                left: -100px;
            }

            .blur-bg-right {
                position: fixed;
                width: 400px;
                height: 400px;
                background: #c084fc;
                filter: blur(150px);
                border-radius: 50%;
                opacity: 0.15;
                z-index: -1;
                bottom: -100px;
                right: -100px;
            }
        </style>
    </head>
    <body>
        <div class="blur-bg"></div>
        <div class="blur-bg-right"></div>
        <div class="container">
            <header>
                <h1>TruthSeeker AI</h1>
                <p class="subtitle">Advanced XGBoost News Classification System</p>
            </header>
            
            <div class="card">
                <textarea id="newsInput" placeholder="Paste full news article text here for analysis..."></textarea>
                <button id="analyzeBtn">
                    <span class="btn-text">Check Credibility</span>
                    <div class="loader" id="loader"></div>
                </button>
                
                <div id="result-container">
                    <div class="result-text" id="resultText"></div>
                </div>
            </div>
        </div>

        <script>
            const btn = document.getElementById('analyzeBtn');
            const input = document.getElementById('newsInput');
            const resultContainer = document.getElementById('result-container');
            const resultText = document.getElementById('resultText');
            const loader = document.getElementById('loader');
            const btnText = document.querySelector('.btn-text');

            btn.addEventListener('click', async () => {
                const text = input.value.trim();
                if (!text) return;

                // Show loading
                loader.style.display = 'block';
                btnText.style.display = 'none';
                btn.disabled = true;
                resultContainer.style.display = 'none';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text })
                    });
                    const data = await response.json();

                    resultContainer.style.display = 'block';
                    resultText.innerText = data.prediction === "Real" ? "✅ THIS IS REAL NEWS" : "⚠️ WARNING: THIS IS FAKE NEWS";
                    
                    resultContainer.className = data.prediction === "Real" ? "result-real" : "result-fake";
                    resultContainer.classList.add('active');

                } catch (e) {
                    alert('Error reaching the analysis engine');
                } finally {
                    loader.style.display = 'none';
                    btnText.style.display = 'block';
                    btn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """

class NewsInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: NewsInput):
    cleaned = preprocess(data.text)
    transform = vectorizer.transform([cleaned])
    pred = model.predict(transform)[0]
    
    # In my training notebook: 1 was Real, 0 was Fake
    # df_true["class"]=1, df_fake["class"]=0
    label = "Real" if pred == 1 else "Fake"
    
    return {"prediction": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

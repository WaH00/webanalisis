from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import io
from pymongo import MongoClient
from collections import Counter

app = Flask(__name__)

# Load model dan vectorizer
model = joblib.load(r"E:\webanalisis\model\naive_bayes_smote_model.joblib")
vectorizer = joblib.load(r"E:\webanalisis\model\tfidf_vectorizer_smote.joblib")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    csv_results = None
    csv_results_all = []
    positive_count = 0
    negative_count = 0

    if request.method == 'POST':
        if 'csv_file' in request.files and request.files['csv_file'].filename != '':
            file = request.files['csv_file']
            try:
                df = pd.read_csv(file)
                if 'review' not in df.columns:
                    csv_results = {'error': "Kolom 'review' tidak ditemukan di file CSV."}
                else:
                    texts = df['review'].astype(str).tolist()
                    vectors = vectorizer.transform(texts)
                    labels = model.predict(vectors)
                    sentiments = ["Positif" if l == 5 else "Negatif" for l in labels]
                    labels = [int(l) for l in labels]
                    csv_results_all = list(zip(texts, labels, sentiments))
                    csv_results = csv_results_all[:10]

                    # ✅ Hitung jumlah Positif & Negatif
                    counter = Counter(sentiments)
                    positive_count = counter.get("Positif", 0)
                    negative_count = counter.get("Negatif", 0)

            except Exception as e:
                csv_results = {'error': f"Gagal membaca file: {e}"}
        else:
            review = request.form.get('review', '')
            if review.strip():
                vector = vectorizer.transform([review])
                label = model.predict(vector)[0]
                sentiment = "Positif" if label == 5 else "Negatif"
                prediction = f"Hasil: {sentiment} (Label: {label})"

    return render_template(
        'index.html',
        prediction=prediction,
        csv_results=csv_results,
        csv_results_all=csv_results_all,
        csv_results_all_len=int(len(csv_results_all)),
        positive_count=positive_count,
        negative_count=negative_count
    )

@app.route('/load_more_csv_results', methods=['POST'])
def load_more_csv_results():
    # Data dikirim ulang dari client (karena tidak ada session/database)
    import json
    data = request.json
    reviews = data.get('reviews', [])
    labels = data.get('labels', [])
    sentiments = data.get('sentiments', [])
    start = int(data.get('start', 0))
    count = int(data.get('count', 10))
    results = list(zip(reviews, labels, sentiments))
    sliced = results[start:start+count]
    return jsonify({'results': sliced})

@app.route('/visualisasi')
def visualisasi():
    return render_template('visualisasi.html')

@app.route('/topwords')
def top_words():
    # Ganti dengan URI MongoDB Atlas Anda
    client = MongoClient("mongodb+srv://wikan:masuk123@analisissentimen.nezv7cl.mongodb.net/?retryWrites=true&w=majority&appName=AnalisisSentimen")
    db = client['SentimentAnalysisDB']
    collection = db['BibitReviewsProcessedTrain']

    # Ambil semua token dari content_no_stopwords
    all_tokens = []
    for doc in collection.find({}, {'content_stemmed': 1}):
        tokens = doc.get('content_stemmed', [])
        if isinstance(tokens, str):
            tokens = tokens.split()
        
        all_tokens.extend(tokens)

    # Hitung frekuensi kata
    counter = Counter(all_tokens)
    top = counter.most_common(10)
    words = [w for w, c in top]
    counts = [c for w, c in top]

    return render_template('perDataset.html', words=words, counts=counts)

    

@app.route('/dataset')
def dataset():
    client = MongoClient("mongodb+srv://wikan:masuk123@analisissentimen.nezv7cl.mongodb.net/?retryWrites=true&w=majority&appName=AnalisisSentime")
    db = client['SentimentAnalysisDB']
    collection = db['BibitReviewsProcessedTrain']

    rows = []
    for doc in collection.find({}, {'content': 1, 'score': 1}).limit(6000):
        content = doc.get('content', '')
        label = doc.get('score', '')
        rows.append((content, label))

    score_1_count = collection.count_documents({'score': 1})
    score_5_count = collection.count_documents({'score': 5})

    return render_template('dataset.html',
                           dataset=rows,
                           scores=[1, 5],
                           score_counts=[score_1_count, score_5_count])


if __name__ == '__main__':
    app.run(debug=True)

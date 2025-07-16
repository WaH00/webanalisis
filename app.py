from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import io
from pymongo import MongoClient
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Stopwords bahasa Indonesia (custom)
stopwords_id = [
    'saya', 'kamu', 'dia', 'itu', 'ini', 'dan', 'atau', 'dengan', 'ke', 'dari',
    'yang', 'untuk', 'pada', 'adalah', 'tidak', 'ya', 'di', 'sebagai', 'akan',
    'oleh', 'karena', 'jadi', 'sudah', 'bisa', 'lagi', 'agar', 'kalau', 'saat',
    'seperti', 'mereka', 'kita', 'apakah', 'namun', 'tersebut', 'semua', 'nya', 
]

app = Flask(__name__)

# Load model dan vectorizer
model = joblib.load(r"E:\webanalisis\model\naive_bayes_smote_model.joblib")
vectorizer = joblib.load(r"E:\webanalisis\model\tfidf_vectorizer_smote.joblib")

# Preprocessing tanpa stemming
def preprocess_text(text):
    tokens = text.lower().split()
    filtered = [word for word in tokens if word not in stopwords_id]
    return ' '.join(filtered)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    csv_results = None
    csv_results_all = []
    positive_count = 0
    negative_count = 0
    avg_length = 0
    min_length = 0
    max_length = 0
    length_by_sentiment = {}
    top_words_positive = []
    top_words_negative = []
    contoh_review_positif = []
    contoh_review_negatif = []
    unique_pos = []
    unique_neg = []
    avg_length_per_sentiment = {}

    if request.method == 'POST':
        if 'csv_file' in request.files and request.files['csv_file'].filename != '':
            file = request.files['csv_file']
            try:
                df = pd.read_csv(file)
                if 'review' not in df.columns:
                    csv_results = {'error': "Kolom 'review' tidak ditemukan di file CSV."}
                else:
                    # Proses prediksi
                    texts = df['review'].astype(str).tolist()
                    vectors = vectorizer.transform(texts)
                    labels = model.predict(vectors)
                    sentiments = ["Positif" if l == 5 else "Negatif" for l in labels]
                    labels = [int(l) for l in labels]
                    csv_results_all = list(zip(texts, labels, sentiments))
                    csv_results = csv_results_all[:10]

                    # Statistik & Struktur
                    counter = Counter(sentiments)
                    positive_count = counter.get("Positif", 0)
                    negative_count = counter.get("Negatif", 0)

                    df_pred = pd.DataFrame(csv_results_all, columns=["review", "label", "sentiment"])
                    df_pred["review_length"] = df_pred["review"].apply(lambda x: len(str(x).split()))
                    avg_length = int(df_pred["review_length"].mean())
                    min_length = int(df_pred["review_length"].min())
                    max_length = int(df_pred["review_length"].max())
                    length_by_sentiment = df_pred.groupby("sentiment")["review_length"].apply(list).to_dict()

                    # ✅ Preprocessing stopword removal
                    df_pred["review_clean"] = df_pred["review"].apply(preprocess_text)

                    # ✅ Top Words Positif
                    df_pos = df_pred[df_pred["sentiment"] == "Positif"]
                    if not df_pos.empty:
                        tfidf_pos_vectorizer = TfidfVectorizer(max_features=1000)
                        tfidf_pos = tfidf_pos_vectorizer.fit_transform(df_pos["review_clean"])
                        sum_pos = tfidf_pos.sum(axis=0).A1
                        words_pos = tfidf_pos_vectorizer.get_feature_names_out()
                        top_pos = sorted(zip(words_pos, sum_pos), key=lambda x: x[1], reverse=True)[:10]
                        top_words_positive = [(w, round(s)) for w, s in top_pos]

                    # ✅ Top Words Negatif
                    df_neg = df_pred[df_pred["sentiment"] == "Negatif"]
                    if not df_neg.empty:
                        tfidf_neg_vectorizer = TfidfVectorizer(max_features=1000)
                        tfidf_neg = tfidf_neg_vectorizer.fit_transform(df_neg["review_clean"])
                        sum_neg = tfidf_neg.sum(axis=0).A1
                        words_neg = tfidf_neg_vectorizer.get_feature_names_out()
                        top_neg = sorted(zip(words_neg, sum_neg), key=lambda x: x[1], reverse=True)[:10]
                        top_words_negative = [(w, round(s)) for w, s in top_neg]

                    # ✅ 2. Contoh Review Positif & Negatif
                    contoh_review_positif = df_pos["review"].head(3).tolist()
                    contoh_review_negatif = df_neg["review"].head(3).tolist()

                    # ✅ 3. Kata Unik
                    words_pos_all = ' '.join(df_pos["review_clean"]).split()
                    words_neg_all = ' '.join(df_neg["review_clean"]).split()
                    set_pos = set(words_pos_all)
                    set_neg = set(words_neg_all)
                    unique_pos = list(set_pos - set_neg)[:10]
                    unique_neg = list(set_neg - set_pos)[:10]

                    # ✅ 4. Rata-rata panjang per sentimen
                    avg_length_per_sentiment = df_pred.groupby("sentiment")["review_length"].mean().round(1).to_dict()

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
        negative_count=negative_count,
        avg_length=avg_length,
        min_length=min_length,
        max_length=max_length,
        length_by_sentiment=length_by_sentiment,
        top_words_positive=top_words_positive,
        top_words_negative=top_words_negative,
        contoh_review_positif=contoh_review_positif,
        contoh_review_negatif=contoh_review_negatif,
        unique_pos=unique_pos,
        unique_neg=unique_neg,
        avg_length_per_sentiment=avg_length_per_sentiment
    )


@app.route('/load_more_csv_results', methods=['POST'])
def load_more_csv_results():
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
    client = MongoClient("mongodb+srv://wikan:masuk123@analisissentimen.nezv7cl.mongodb.net/?retryWrites=true&w=majority&appName=AnalisisSentimen")
    db = client['SentimentAnalysisDB']
    collection = db['BibitReviewsProcessedTrain']

    all_tokens = []
    for doc in collection.find({}, {'content_stemmed': 1}):
        tokens = doc.get('content_stemmed', [])
        if isinstance(tokens, str):
            tokens = tokens.split()
        all_tokens.extend(tokens)

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

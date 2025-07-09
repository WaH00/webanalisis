from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import io

app = Flask(__name__)

# Load model dan vectorizer
model = joblib.load(r"E:\webanalisis\model\naive_bayes_smote_model.joblib")
vectorizer = joblib.load(r"E:\webanalisis\model\tfidf_vectorizer_smote.joblib")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    csv_results = None
    csv_results_all = []
    if request.method == 'POST':
        # Cek apakah upload file atau input manual
        if 'csv_file' in request.files and request.files['csv_file'].filename != '':
            file = request.files['csv_file']
            try:
                df = pd.read_csv(file)
                # Asumsi kolom review bernama 'review'
                if 'review' not in df.columns:
                    csv_results = {'error': "Kolom 'review' tidak ditemukan di file CSV."}
                else:
                    texts = df['review'].astype(str).tolist()
                    vectors = vectorizer.transform(texts)
                    labels = model.predict(vectors)
                    sentiments = ["Positif" if l == 5 else "Negatif" for l in labels]
                    # Konversi labels ke int
                    labels = [int(l) for l in labels]
                    csv_results_all = list(zip(texts, labels, sentiments))
                    csv_results = csv_results_all[:10]  # hanya 10 pertama
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
        csv_results_all_len=int(len(csv_results_all))  # pastikan int asli
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

if __name__ == '__main__':
    app.run(debug=True)

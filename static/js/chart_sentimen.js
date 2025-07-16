export function renderSentimentChart(positiveCount, negativeCount) {
    const ctx = document.getElementById('sentimentChart')?.getContext('2d');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Negatif', 'Positif'],
            datasets: [{
                label: 'Jumlah',
                data: [negativeCount, positiveCount],
                backgroundColor: ['#f44336', '#4caf50']
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

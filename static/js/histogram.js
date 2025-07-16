export function renderLengthHistogramChart(posData, negData) {
    const ctx = document.getElementById('lengthHistogramChart')?.getContext('2d');
    if (!ctx) return;

    function getHistogramData(data, binSize = 5) {
        const max = Math.max(...data);
        const bins = Array(Math.ceil(max / binSize)).fill(0);
        data.forEach(val => {
            const index = Math.floor(val / binSize);
            bins[index] = (bins[index] || 0) + 1;
        });
        return bins;
    }

    const binLabels = [...Array(20).keys()].map(i => `${i * 5}-${i * 5 + 4}`);
    const posHist = getHistogramData(posData);
    const negHist = getHistogramData(negData);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [
                {
                    label: 'Positif',
                    data: posHist,
                    backgroundColor: '#4caf50'
                },
                {
                    label: 'Negatif',
                    data: negHist,
                    backgroundColor: '#f44336'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'top' } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

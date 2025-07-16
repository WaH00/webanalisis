import { renderLengthHistogramChart } from './histogram.js';
import { renderSentimentChart } from './chart_sentimen.js';

document.addEventListener('DOMContentLoaded', () => {
    if (window.sentimentCounts) {
        const { positive, negative } = window.sentimentCounts;
        renderSentimentChart(positive, negative);
    }

    if (window.lengthData) {
        const { pos, neg } = window.lengthData;
        renderLengthHistogramChart(pos, neg);
    }
});



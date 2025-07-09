document.addEventListener('DOMContentLoaded', function () {
    // Slider untuk mengatur tinggi box hasil
    const slider = document.getElementById('box-height-slider');
    const box = document.getElementById('csv-result-box');
    const value = document.getElementById('box-height-value');
    if (slider && box && value) {
        slider.addEventListener('input', function() {
            box.style.maxHeight = this.value + 'px';
            value.textContent = this.value + 'px';
        });
    }

    // Load More untuk hasil CSV
    const loadMoreBtn = document.getElementById('load-more-btn');
    if (loadMoreBtn && window.allCsvResults && Array.isArray(window.allCsvResults) && window.csvResultsAllLen) {
        let loadedCount = 10;
        loadMoreBtn.onclick = function() {
            const nextRows = window.allCsvResults.slice(loadedCount, loadedCount + 10);
            let table = document.getElementById('csv-result-table');
            nextRows.forEach(function(row, idx) {
                let tr = document.createElement('tr');
                tr.innerHTML = `<td>${loadedCount + idx + 1}</td>
                                <td style="text-align:left;">${row[0]}</td>
                                <td>${row[1]}</td>
                                <td>${row[2]}</td>`;
                table.appendChild(tr);
            });
            loadedCount += nextRows.length;
            if (loadedCount >= window.csvResultsAllLen) {
                loadMoreBtn.style.display = 'none';
            }
        };
    }
});

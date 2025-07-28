document.addEventListener('DOMContentLoaded', () => {
    const newsArticleInput = document.getElementById('newsArticle');
    const predictButton = document.getElementById('predictButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const predictionResults = document.getElementById('predictionResults');
    const lrResult = document.getElementById('lrResult');
    const dtcResult = document.getElementById('dtcResult');
    const rfcResult = document.getElementById('rfcResult');
    const gbcResult = document.getElementById('gbcResult');
    const errorMessageDiv = document.getElementById('errorMessage');

    const hideAllResults = () => {
        lrResult.classList.add('hidden');
        dtcResult.classList.add('hidden');
        rfcResult.classList.add('hidden');
        gbcResult.classList.add('hidden');
        errorMessageDiv.classList.add('hidden');
    };

    const updateResultDisplay = (element, prediction) => {
        element.classList.remove('hidden');
        element.querySelector('.result-text').textContent = prediction;
        if (prediction === 'FAKE NEWS') {
            element.className = 'bg-red-50 p-4 rounded-lg border border-red-200 text-red-800 font-medium text-lg';
        } else {
            element.className = 'bg-green-50 p-4 rounded-lg border border-green-200 text-green-800 font-medium text-lg';
        }
    };

    predictButton.addEventListener('click', async () => {
        const newsArticle = newsArticleInput.value.trim();

        if (newsArticle === '') {
            hideAllResults();
            errorMessageDiv.classList.remove('hidden');
            errorMessageDiv.querySelector('.error-message-text').textContent = 'Please enter a news article to analyze.';
            return;
        }

        hideAllResults();
        loadingIndicator.classList.remove('hidden');
        predictButton.disabled = true;

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ news_article: newsArticle }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            updateResultDisplay(lrResult, data.LR_Prediction);
            updateResultDisplay(dtcResult, data.DTC_Prediction);
            updateResultDisplay(rfcResult, data.RFC_Prediction);
            updateResultDisplay(gbcResult, data.GBC_Prediction);

        } catch (error) {
            console.error('Error during prediction:', error);
            hideAllResults();
            errorMessageDiv.classList.remove('hidden');
            errorMessageDiv.querySelector('.error-message-text').textContent = `Failed to get prediction: ${error.message}. Please ensure the Python backend is running and accessible.`;
        } finally {
            loadingIndicator.classList.add('hidden');
            predictButton.disabled = false;
        }
    });
});

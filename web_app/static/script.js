/**
 * AI Bankruptcy Risk Prediction - Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnSpinner = submitBtn.querySelector('.btn-spinner');
    const formErrorMessage = document.getElementById('formErrorMessage');

    // Result Section Elements
    const resultSection = document.getElementById('resultSection');
    const resultCard = document.getElementById('resultCard');
    const resultDecision = document.getElementById('resultDecision');
    const riskBadge = document.getElementById('riskBadge');
    const riskPercentage = document.getElementById('riskPercentage');
    const progressFill = document.getElementById('progressFill');
    const rawProbability = document.getElementById('rawProbability');
    const classificationStatus = document.getElementById('classificationStatus');

    // Quick Action Buttons
    const btnLoadHealthy = document.getElementById('btnLoadHealthy');
    const btnLoadBankrupt = document.getElementById('btnLoadBankrupt');
    const btnLoadMean = document.getElementById('btnLoadMean');
    const btnClearForm = document.getElementById('btnClearForm');

    // Search and Filter
    const searchInput = document.getElementById('featureSearch');
    const featureCountDisplay = document.getElementById('featureCountDisplay');
    const inputCards = document.querySelectorAll('.input-card');
    const allInputs = document.querySelectorAll('.feature-input');

    // Real-time input validation cleanup
    allInputs.forEach(input => {
        input.addEventListener('input', () => {
            if (input.value.trim() !== '' && !isNaN(Number(input.value))) {
                input.closest('.input-card').classList.remove('has-error');
            }
        });
    });

    // -------------------------------------------------------------------------
    // Form Submission & Prediction
    // -------------------------------------------------------------------------
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();

        // 1. Client-side Validation of all 95 features
        let hasErrors = false;
        let firstInvalidCard = null;
        let emptyCount = 0;
        const payloadData = {};

        allInputs.forEach(input => {
            const card = input.closest('.input-card');
            const rawVal = input.value.trim();
            const featureName = input.getAttribute('name');

            if (rawVal === '') {
                hasErrors = true;
                emptyCount++;
                card.classList.add('has-error');
                if (!firstInvalidCard) firstInvalidCard = card;
            } else {
                const numVal = Number(rawVal);
                if (isNaN(numVal) || !isFinite(numVal)) {
                    hasErrors = true;
                    card.classList.add('has-error');
                    if (!firstInvalidCard) firstInvalidCard = card;
                } else {
                    card.classList.remove('has-error');
                    payloadData[featureName] = numVal;
                }
            }
        });

        if (hasErrors) {
            showError(`Please provide valid numeric values for all 95 features. (${emptyCount} empty or invalid fields)`);
            if (firstInvalidCard) {
                // If the card is hidden by search filter, clear search first
                if (firstInvalidCard.style.display === 'none') {
                    searchInput.value = '';
                    filterFeatures('');
                }
                firstInvalidCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                const errInput = firstInvalidCard.querySelector('.feature-input');
                if (errInput) errInput.focus();
            }
            return;
        }

        // 2. Submit to Backend API
        setLoading(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ features: payloadData })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Prediction failed. Please check your inputs.');
            }

            // 3. Render Prediction Report
            displayResults(data);

        } catch (err) {
            console.error('Prediction Error:', err);
            showError(err.message || 'An unexpected error occurred while communicating with the model.');
        } finally {
            setLoading(false);
        }
    });

    // -------------------------------------------------------------------------
    // Render Results
    // -------------------------------------------------------------------------
    function displayResults(data) {
        // data format: { prediction: 0 or 1, probability: 0.8734, percentage: 87.34 }
        const { prediction, probability, percentage } = data;

        // Reset state classes
        resultCard.classList.remove('is-bankrupt', 'is-non-bankrupt');

        if (prediction === 1) {
            resultCard.classList.add('is-bankrupt');
            resultDecision.textContent = 'Bankrupt';
            riskBadge.textContent = 'High Bankruptcy Risk';
            classificationStatus.textContent = 'Bankrupt (Class 1)';
        } else {
            resultCard.classList.add('is-non-bankrupt');
            resultDecision.textContent = 'Non-Bankrupt';
            riskBadge.textContent = 'Low Risk / Financially Stable';
            classificationStatus.textContent = 'Non-Bankrupt (Class 0)';
        }

        riskPercentage.textContent = percentage.toFixed(2);
        rawProbability.textContent = probability.toFixed(4);

        // Update progress bar
        const clampedPct = Math.min(Math.max(percentage, 0), 100);
        progressFill.style.width = `${clampedPct}%`;

        // Show section & scroll
        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // -------------------------------------------------------------------------
    // Helper & Loading UI
    // -------------------------------------------------------------------------
    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.disabled = true;
            btnSpinner.style.display = 'inline-block';
            btnText.textContent = 'EVALUATING MODEL...';
        } else {
            submitBtn.disabled = false;
            btnSpinner.style.display = 'none';
            btnText.textContent = 'PREDICT BANKRUPTCY RISK';
        }
    }

    function showError(msg) {
        formErrorMessage.textContent = msg;
        formErrorMessage.style.display = 'block';
        formErrorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideError() {
        formErrorMessage.style.display = 'none';
        formErrorMessage.textContent = '';
    }

    // -------------------------------------------------------------------------
    // Quick Test Sample Data Loaders
    // -------------------------------------------------------------------------
    async function loadSample(sampleType) {
        hideError();
        setLoading(true);
        try {
            const res = await fetch(`/api/sample?type=${sampleType}`);
            if (!res.ok) throw new Error('Failed to retrieve sample data.');
            const data = await res.json();
            const sampleFeatures = data.features;

            allInputs.forEach(input => {
                const name = input.getAttribute('name');
                if (sampleFeatures[name] !== undefined) {
                    input.value = sampleFeatures[name];
                }
                input.closest('.input-card').classList.remove('has-error');
            });

        } catch (err) {
            showError('Could not load sample data: ' + err.message);
        } finally {
            setLoading(false);
        }
    }

    btnLoadHealthy.addEventListener('click', () => loadSample('non_bankrupt'));
    btnLoadBankrupt.addEventListener('click', () => loadSample('bankrupt'));
    btnLoadMean.addEventListener('click', () => loadSample('mean'));

    btnClearForm.addEventListener('click', () => {
        allInputs.forEach(input => {
            input.value = '';
            input.closest('.input-card').classList.remove('has-error');
        });
        hideError();
        resultSection.style.display = 'none';
    });

    // -------------------------------------------------------------------------
    // Search / Filter through 95 features
    // -------------------------------------------------------------------------
    function filterFeatures(query) {
        const lowerQ = query.trim().toLowerCase();
        let visibleCount = 0;

        inputCards.forEach(card => {
            const featName = card.getAttribute('data-feature-name') || '';
            if (lowerQ === '' || featName.includes(lowerQ)) {
                card.style.display = 'flex';
                visibleCount++;
            } else {
                card.style.display = 'none';
            }
        });

        featureCountDisplay.textContent = `Showing ${visibleCount} of ${inputCards.length}`;
    }

    searchInput.addEventListener('input', (e) => {
        filterFeatures(e.target.value);
    });
});

// Main JavaScript file for Fake News Detector

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const newsForm = document.getElementById('news-form');
    const articleTitle = document.getElementById('article-title');
    const articleText = document.getElementById('article-text');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultSection = document.getElementById('result-section');
    const resultHeader = document.getElementById('result-header');
    const resultTitle = document.getElementById('result-title');
    const resultIcon = document.getElementById('result-icon');
    const resultLabel = document.getElementById('result-label');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceScore = document.getElementById('confidence-score');
    const explanationText = document.getElementById('explanation-text');

    // Sample article data (optional functionality)
    const sampleArticles = [
        {
            title: "Scientists discover new planet in solar system",
            text: "In a groundbreaking discovery, astronomers have identified a previously unknown planet orbiting our sun. The celestial body, temporarily named 'Planet X', was detected using advanced imaging technology from the Hubble Space Telescope. Researchers estimate it has a mass approximately 3 times that of Earth and orbits beyond Neptune. This finding challenges our current understanding of our solar system's formation and evolution. The international team of scientists will publish their full findings in next month's issue of Nature."
        },
        {
            title: "Man grows third arm after COVID vaccine",
            text: "A shocking report has emerged claiming that a 45-year-old man from Minnesota has grown a fully functional third arm after receiving his COVID-19 vaccine. The man, who wishes to remain anonymous, claims the extra limb began developing just 3 days after his second dose. Medical experts are baffled by this unprecedented side effect. The government has allegedly been suppressing similar reports from around the country. The man is now considering legal action against the vaccine manufacturer, while also enjoying his newfound ability to multitask."
        }
    ];

    // Event listeners
    newsForm.addEventListener('submit', handleFormSubmit);
    clearBtn.addEventListener('click', clearForm);

    // Function to handle form submission
    async function handleFormSubmit(e) {
        e.preventDefault();
        
        // Validate input
        const title = articleTitle.value.trim();
        const text = articleText.value.trim();
        
        if (!text) {
            showAlert('Please enter the article text', 'danger');
            return;
        }
        
        // Show loading state
        showLoading();
        
        try {
            // Send data to API
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title: title,
                    text: text
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Show result
                displayResult(data.result);
            } else {
                // Show error
                showAlert(`Error: ${data.error}`, 'danger');
                hideResultSection();
            }
        } catch (error) {
            console.error('Error:', error);
            showAlert('Server error. Please try again later.', 'danger');
            hideResultSection();
        }
    }
    
    // Function to display the prediction result
    function displayResult(result) {
        // Calculate percentage for display
        const confidencePercentage = Math.round(result.confidence * 100);
        
        // Update UI elements
        if (result.label === 'FAKE') {
            resultHeader.className = 'card-header bg-danger text-white';
            resultTitle.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Fake News Detected';
            resultIcon.innerHTML = '<i class="fas fa-times-circle text-danger"></i>';
            resultLabel.textContent = 'This article is likely FAKE';
            confidenceBar.className = 'progress-bar bg-danger';
            explanationText.innerHTML = 'Our analysis indicates this article contains patterns commonly found in fake news. Be cautious about sharing this content without verifying from reliable sources.';
        } else {
            resultHeader.className = 'card-header bg-success text-white';
            resultTitle.innerHTML = '<i class="fas fa-check-circle me-2"></i>Authentic Content';
            resultIcon.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
            resultLabel.textContent = 'This article appears to be REAL';
            confidenceBar.className = 'progress-bar bg-success';
            explanationText.innerHTML = 'Our analysis suggests this article follows patterns consistent with legitimate news. However, always practice critical thinking and verify important information.';
        }
        
        // Update confidence indicators
        confidenceBar.style.width = `${confidencePercentage}%`;
        confidenceScore.textContent = confidencePercentage;
        
        // Show result section
        resultSection.style.display = 'block';
        
        // Scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to clear the form
    function clearForm() {
        articleTitle.value = '';
        articleText.value = '';
        hideResultSection();
        articleTitle.focus();
    }
    
    // Function to hide result section
    function hideResultSection() {
        resultSection.style.display = 'none';
    }
    
    // Function to show loading state
    function showLoading() {
        resultSection.style.display = 'block';
        resultHeader.className = 'card-header bg-primary text-white';
        resultTitle.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing';
        resultIcon.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';
        resultLabel.textContent = 'Processing your article...';
        confidenceBar.style.width = '0%';
        confidenceScore.textContent = '0';
        explanationText.innerHTML = 'Our AI model is analyzing the text patterns, language structure, and content of your article. This should take only a few seconds.';
        
        // Scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to show alert message
    function showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert alert before the form
        newsForm.parentNode.insertBefore(alertDiv, newsForm);
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }

    // Optional: Add sample article functionality
    // You can add buttons in the HTML and connect them to this function
    function loadSampleArticle(index) {
        if (sampleArticles[index]) {
            articleTitle.value = sampleArticles[index].title;
            articleText.value = sampleArticles[index].text;
            hideResultSection();
        }
    }

    // Optional: Export functions for testing or external access
    window.FakeNewsDetector = {
        loadSampleArticle
    };
});
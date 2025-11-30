document.addEventListener('DOMContentLoaded', function() {
    // Handle intro animation
    const introOverlay = document.getElementById('intro-animation');
    if (introOverlay) {
        // Hide intro animation after 3.5 seconds
        setTimeout(function() {
            introOverlay.style.opacity = '0';
            introOverlay.style.visibility = 'hidden';
            
            // Enable scrolling on body
            document.body.style.overflow = 'auto';
        }, 3500);
    }

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
    if (newsForm) newsForm.addEventListener('submit', handleFormSubmit);
    if (clearBtn) clearBtn.addEventListener('click', clearForm);
    // Add direct event listener for analyze button as a backup
    if (analyzeBtn) analyzeBtn.addEventListener('click', function(e) {
        e.preventDefault();
        if (newsForm) {
            // Trigger the form submission
            const submitEvent = new Event('submit', { cancelable: true });
            newsForm.dispatchEvent(submitEvent);
        }
    });
    
    // Initialize tab functionality
    initializeTabs();
    
    // Initialize sample article buttons
    initializeSampleArticles();
    
    // Initialize stats counter animation
    animateStats();
    
    // Initialize floating elements animation
    initializeFloatingElements();
    
    // Initialize navbar scroll effect
    initializeNavbarScroll();
    
    // Initialize smooth scrolling for navigation links
    initializeSmoothScrolling();
    
    // Initialize share buttons
    initializeShareButtons();
    
    // Initialize hero buttons
    initializeHeroButtons();

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
            const response = await fetch('http://localhost:5000/api/predict', {
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
        if (resultSection) resultSection.style.display = 'none';
    }
    
    // Function to show loading state
    function showLoading() {
        if (!resultSection) return;
        
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
        if (!newsForm) return;
        
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

    // Function to initialize tabs
    function initializeTabs() {
        const tabs = document.querySelectorAll('.tab');
        if (tabs.length > 0) {
            tabs.forEach(tab => {
                tab.addEventListener('click', function() {
                    // Remove active class from all tabs
                    tabs.forEach(t => t.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    this.classList.add('active');
                    
                    // Hide all tab content
                    const tabContents = document.querySelectorAll('.tab-content');
                    tabContents.forEach(content => content.classList.remove('active'));
                    
                    // Show selected tab content
                    const tabId = this.getAttribute('data-tab');
                    const targetContent = document.getElementById(tabId + '-content');
                    if (targetContent) targetContent.classList.add('active');
                });
            });
        }
    }
    
    // Function to initialize sample article buttons
    function initializeSampleArticles() {
        const sampleButtons = document.querySelectorAll('.load-sample');
        if (sampleButtons.length > 0) {
            sampleButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const sampleArticle = this.closest('.sample-article');
                    if (sampleArticle) {
                        const index = parseInt(sampleArticle.getAttribute('data-index'));
                        loadSampleArticle(index);
                        
                        // Switch to manual input tab
                        const manualTab = document.querySelector('[data-tab="manual-input"]');
                        if (manualTab) {
                            manualTab.click();
                        }
                    }
                });
            });
        }
    }
    
    // Function to animate stats
    function animateStats() {
        const statNumbers = document.querySelectorAll('.stat-number');
        if (statNumbers.length > 0) {
            statNumbers.forEach(stat => {
                const targetValue = parseFloat(stat.getAttribute('data-val') || '0');
                const duration = 2000; // 2 seconds
                const startTime = Date.now();
                const startValue = 0;
                
                function updateCounter() {
                    const currentTime = Date.now();
                    const elapsedTime = currentTime - startTime;
                    
                    if (elapsedTime < duration) {
                        const progress = elapsedTime / duration;
                        const currentValue = startValue + (targetValue - startValue) * progress;
                        stat.textContent = targetValue >= 100 ? 
                            Math.floor(currentValue) : 
                            currentValue.toFixed(1);
                        requestAnimationFrame(updateCounter);
                    } else {
                        stat.textContent = targetValue;
                    }
                }
                
                requestAnimationFrame(updateCounter);
            });
        }
    }

    // Function to load sample article
    function loadSampleArticle(index) {
        if (sampleArticles[index] && articleTitle && articleText) {
            articleTitle.value = sampleArticles[index].title;
            articleText.value = sampleArticles[index].text;
            hideResultSection();
        }
    }
    
    // Function to initialize floating elements animation
    function initializeFloatingElements() {
        const floatingIcons = document.querySelectorAll('.floating-icon');
        if (floatingIcons.length > 0) {
            floatingIcons.forEach((icon, index) => {
                // Set random initial positions
                const randomX = Math.random() * 20 - 10; // -10 to 10
                const randomY = Math.random() * 20 - 10; // -10 to 10
                
                // Set animation properties
                icon.style.animation = `float ${3 + index * 0.5}s ease-in-out infinite alternate`;
                icon.style.transform = `translate(${randomX}px, ${randomY}px)`;
            });
        }
    }
    
    // Function to initialize navbar scroll effect
    function initializeNavbarScroll() {
        const navbar = document.querySelector('.navbar');
        if (navbar) {
            window.addEventListener('scroll', function() {
                if (window.scrollY > 50) {
                    navbar.classList.add('scrolled');
                } else {
                    navbar.classList.remove('scrolled');
                }
            });
        }
    }
    
    // Function to initialize smooth scrolling for navigation links
    function initializeSmoothScrolling() {
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
        if (navLinks.length > 0) {
            navLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    // Get the target section id from href
                    const targetId = this.getAttribute('href');
                    if (targetId && targetId.startsWith('#') && targetId.length > 1) {
                        e.preventDefault();
                        
                        // Get the target element
                        const targetElement = document.querySelector(targetId);
                        if (targetElement) {
                            // Scroll to the target element
                            targetElement.scrollIntoView({
                                behavior: 'smooth'
                            });
                            
                            // Close mobile menu if open
                            const navbarCollapse = document.querySelector('.navbar-collapse');
                            if (navbarCollapse && navbarCollapse.classList.contains('show')) {
                                navbarCollapse.classList.remove('show');
                            }
                            
                            // Update active nav link
                            navLinks.forEach(navLink => navLink.classList.remove('active'));
                            this.classList.add('active');
                        }
                    }
                });
            });
        }
    }
    
    // Function to initialize share buttons
    function initializeShareButtons() {
        const shareButtons = document.querySelectorAll('.share-buttons .btn-social');
        if (shareButtons.length > 0) {
            shareButtons.forEach((button, index) => {
                button.addEventListener('click', function() {
                    // Get result information
                    const resultText = resultLabel ? resultLabel.textContent : '';
                    const confidenceValue = confidenceScore ? confidenceScore.textContent : '0';
                    const shareText = `I just analyzed an article with FakeGuard AI. Result: ${resultText} (${confidenceValue}% confidence)`;
                    const shareUrl = window.location.href;
                    
                    // Share on different platforms
                    let shareLink = '';
                    
                    switch(index) {
                        case 0: // Twitter
                            shareLink = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`;
                            break;
                        case 1: // Facebook
                            shareLink = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}&quote=${encodeURIComponent(shareText)}`;
                            break;
                        case 2: // LinkedIn
                            shareLink = `https://www.linkedin.com/shareArticle?mini=true&url=${encodeURIComponent(shareUrl)}&title=Fake News Detection&summary=${encodeURIComponent(shareText)}`;
                            break;
                        case 3: // Email
                            shareLink = `mailto:?subject=Fake News Detection Result&body=${encodeURIComponent(shareText + '\n\n' + shareUrl)}`;
                            break;
                    }
                    
                    // Open share dialog
                    if (shareLink) {
                        window.open(shareLink, '_blank');
                    }
                });
            });
        }
    }
    
    // Function to update active navigation based on scroll position
    function updateActiveNavOnScroll() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
        
        if (sections.length > 0 && navLinks.length > 0) {
            window.addEventListener('scroll', function() {
                let current = '';
                
                sections.forEach(section => {
                    const sectionTop = section.offsetTop - 100;
                    const sectionHeight = section.offsetHeight;
                    
                    if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                        current = '#' + section.getAttribute('id');
                    }
                });
                
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === current) {
                        link.classList.add('active');
                    }
                });
            });
        }
    }
    
    // Function to initialize hero buttons (Start Analysis and Learn More)
    function initializeHeroButtons() {
        // Handle Start Analysis button - Fix the selector to target the correct button
        const startAnalysisBtn = document.querySelector('.hero-buttons .btn-gradient.primary');
        if (startAnalysisBtn) {
            startAnalysisBtn.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Scroll to the analysis section
                const analysisSection = document.getElementById('analyzer');
                if (analysisSection) {
                    analysisSection.scrollIntoView({ behavior: 'smooth' });
                    
                    // Focus on the article title input after scrolling
                    setTimeout(() => {
                        if (articleTitle) {
                            articleTitle.focus();
                        }
                    }, 800);
                }
            });
        }
        
        // Handle Learn More button - Fix the selector to target the correct button
        const learnMoreBtn = document.querySelector('.hero-buttons .btn-gradient.secondary');
        if (learnMoreBtn) {
            learnMoreBtn.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Scroll to the about section
                const aboutSection = document.getElementById('about');
                if (aboutSection) {
                    aboutSection.scrollIntoView({ behavior: 'smooth' });
                }
            });
        }
    }
    
    // Initialize scroll-based navigation highlighting
    updateActiveNavOnScroll();

    // Optional: Export functions for testing or external access
    window.FakeNewsDetector = {
        loadSampleArticle,
        clearForm,
        showAlert
    };
});
// Theme Toggle
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

themeToggle.addEventListener('click', () => {
    body.classList.toggle('dark');
    const icon = themeToggle.querySelector('i');
    icon.classList.toggle('fa-moon');
    icon.classList.toggle('fa-sun');
});

// Language Toggle
const langToggle = document.getElementById('langToggle');
let currentLang = 'en';

const translations = {
    en: {
        nav: ['Home', 'Features', 'About', 'Contact', 'Get Started'],
        hero: {
            title: 'Smart Farming for Better Yields',
            subtitle: 'Make data-driven decisions with AI-powered tools designed specifically for farmers. Increase yields, predict prices, and protect your crops.',
        },
        features: {
            title: 'Intelligent Tools for Modern Farmers',
            subtitle: 'Our AI-powered tools help you make informed decisions at every step of your farming journey'
        },
        about: {
            title: 'Powered by Advanced AI',
            subtitle: 'Leveraging cutting-edge technology for smarter farming',
            content: [
                'Our machine learning algorithms analyze historical data, weather patterns, and market trends to provide accurate price predictions for your crops.',
                'Advanced computer vision and deep learning models help detect plant diseases with over 95% accuracy.',
                'Smart recommendation systems consider soil conditions, climate data, and market demand to suggest the most profitable crops.'
            ]
        },
        contact: {
            title: 'Get in Touch',
            subtitle: 'We\'re here to help you succeed',
            form: {
                name: 'Your Name',
                email: 'Your Email',
                message: 'Your Message',
                submit: 'Send Message'
            }
        }
    },
    hi: {
        nav: ['होम', 'सुविधाएं', 'हमारे बारे में', 'संपर्क करें', 'शुरू करें'],
        hero: {
            title: 'बेहतर उपज के लिए स्मार्ट खेती',
            subtitle: 'किसानों के लिए विशेष रूप से डिज़ाइन किए गए AI-संचालित टूल्स के साथ डेटा-आधारित निर्णय लें। उपज बढ़ाएं, कीमतों की भविष्यवाणी करें और अपनी फसलों की रक्षा करें।',
        },
        features: {
            title: 'आधुनिक किसानों के लिए बुद्धिमान उपकरण',
            subtitle: 'हमारे AI-संचालित टूल आपकी खेती यात्रा के हर कदम पर सूचित निर्णय लेने में मदद करते हैं'
        },
        about: {
            title: 'उन्नत एआई द्वारा संचालित',
            subtitle: 'स्मार्ट खेती के लिए अत्याधुनिक तकनीक का उपयोग',
            content: [
                'हमारे मशीन लर्निंग एल्गोरिदम ऐतिहासिक डेटा, मौसम पैटर्न और बाजार के रुझानों का विश्लेषण करके आपकी फसलों के लिए सटीक मूल्य भविष्यवाणी प्रदान करते हैं।',
                'उन्नत कंप्यूटर विजन और डीप लर्निंग मॉडल 95% से अधिक सटीकता के साथ पौधों की बीमारियों का पता लगाने में मदद करते हैं।',
                'स्मार्ट अनुशंसा प्रणालियां सबसे लाभदायक फसलों का सुझाव देने के लिए मिट्टी की स्थिति, जलवायु डेटा और बाजार की मांग पर विचार करती हैं।'
            ]
        },
        contact: {
            title: 'संपर्क करें',
            subtitle: 'हम आपकी सफलता में सहायता के लिए यहां हैं',
            form: {
                name: 'आपका नाम',
                email: 'आपका ईमेल',
                message: 'आपका संदेश',
                submit: 'संदेश भेजें'
            }
        }
    }
};

langToggle.addEventListener('click', () => {
    currentLang = currentLang === 'en' ? 'hi' : 'en';
    langToggle.textContent = currentLang.toUpperCase();
    updateLanguage();
});

function updateLanguage() {
    document.querySelectorAll('[data-en]').forEach(element => {
        const key = currentLang === 'en' ? 'data-en' : 'data-hi';
        element.textContent = element.getAttribute(key);
    });

    // Update navigation
    const navLinks = document.querySelectorAll('.nav-links a');
    translations[currentLang].nav.forEach((text, index) => {
        if (navLinks[index]) navLinks[index].textContent = text;
    });

    // Update form placeholders
    const formTranslations = translations[currentLang].contact.form;
    document.querySelector('#contact-name').placeholder = formTranslations.name;
    document.querySelector('#contact-email').placeholder = formTranslations.email;
    document.querySelector('#contact-message').placeholder = formTranslations.message;
    document.querySelector('#contact-submit').textContent = formTranslations.submit;
}

// Smooth scroll to sections
document.querySelectorAll('nav a').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const section = link.textContent.toLowerCase();
        const element = document.getElementById(section);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Chatbot functionality (unchanged)
const chatbotToggle = document.querySelector('.chatbot-toggle');
const chatbotContainer = document.querySelector('.chatbot-container');
const chatbotMessages = document.querySelector('.chatbot-messages');
const chatbotInput = document.querySelector('.chatbot-input input');
const chatbotSend = document.querySelector('.chatbot-input button');
const closeChat = document.querySelector('.close-chat');

const botResponses = {
    greeting: "Hello! I'm your KrishiGuide assistant. How can I help you today?",
    default: "I understand you need help. Please specify your question about crop suggestions, price predictions, disease detection, or weather forecasts.",
    keywords: {
        crop: "I can help you with crop suggestions based on your soil type and climate. Would you like to know more?",
        price: "I can provide price predictions for various crops based on market trends. Which crop are you interested in?",
        disease: "I can help identify plant diseases. Please upload a photo of your affected crop.",
        weather: "I can provide detailed weather forecasts for your farm location. Please share your location.",
    }
};

function addMessage(message, isBot = false) {
    const messageDiv = document.createElement('div');
    messageDiv.style.marginBottom = '10px';
    messageDiv.style.padding = '8px 12px';
    messageDiv.style.borderRadius = '10px';
    messageDiv.style.maxWidth = '80%';
    messageDiv.style.wordWrap = 'break-word';
    
    if (isBot) {
        messageDiv.style.backgroundColor = '#4CAF50';
        messageDiv.style.color = 'white';
        messageDiv.style.marginRight = 'auto';
    } else {
        messageDiv.style.backgroundColor = '#e9ecef';
        messageDiv.style.color = '#333';
        messageDiv.style.marginLeft = 'auto';
    }
    
    messageDiv.textContent = message;
    chatbotMessages.appendChild(messageDiv);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
}

chatbotToggle.addEventListener('click', () => {
    chatbotContainer.classList.toggle('hidden');
    if (!chatbotContainer.classList.contains('hidden') && chatbotMessages.children.length === 0) {
        addMessage(botResponses.greeting, true);
    }
});

closeChat.addEventListener('click', () => {
    chatbotContainer.classList.add('hidden');
});

chatbotSend.addEventListener('click', sendMessage);
chatbotInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const message = chatbotInput.value.trim();
    if (message) {
        addMessage(message);
        chatbotInput.value = '';
        
        setTimeout(() => {
            let response = botResponses.default;
            for (const [keyword, reply] of Object.entries(botResponses.keywords)) {
                if (message.toLowerCase().includes(keyword)) {
                    response = reply;
                    break;
                }
            }
            addMessage(response, true);
        }, 1000);
    }
}
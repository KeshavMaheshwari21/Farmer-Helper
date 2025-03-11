// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const body = document.body;
let isDark = false;

themeToggle.addEventListener('click', () => {
    isDark = !isDark;
    body.className = isDark ? 'dark' : 'light';
    themeToggle.textContent = isDark ? '🌚' : '🌞';
});

// Language toggle functionality
const languageToggle = document.getElementById('languageToggle');
let isHindi = false;

function updateLanguage() {
    const elements = document.querySelectorAll('[data-en]');
    elements.forEach(element => {
        element.textContent = isHindi ? element.getAttribute('data-hi') : element.getAttribute('data-en');
    });
    languageToggle.textContent = isHindi ? 'हिं' : 'EN';
}

languageToggle.addEventListener('click', () => {
    isHindi = !isHindi;
    updateLanguage();
});

// Chatbot functionality
const chatbotIcon = document.getElementById('chatbotIcon');
const chatWindow = document.getElementById('chatWindow');
const closeChat = document.getElementById('closeChat');
const messageInput = document.getElementById('messageInput');
const sendMessage = document.getElementById('sendMessage');

chatbotIcon.addEventListener('click', () => {
    chatWindow.classList.add('active');
});

closeChat.addEventListener('click', () => {
    chatWindow.classList.remove('active');
});

function addMessage(message, isUser = false) {
    const messagesContainer = document.querySelector('.chat-messages');
    const messageElement = document.createElement('div');
    messageElement.className = `message ${isUser ? 'user' : 'bot'}`;
    messageElement.innerHTML = `<p>${message}</p>`;
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function handleUserMessage() {
    const message = messageInput.value.trim();
    if (message) {
        addMessage(message, true);
        messageInput.value = '';
        
        // Simulate bot response
        setTimeout(() => {
            const response = "Thank you for your message. Our AI assistant will help you with your farming queries.";
            addMessage(response);
        }, 1000);
    }
}

sendMessage.addEventListener('click', handleUserMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleUserMessage();
    }
});
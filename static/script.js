// ---------------------- Theme Toggle ----------------------
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

// Load theme from localStorage
const savedTheme = localStorage.getItem('theme') || 'light';
if (savedTheme === 'dark') {
    body.classList.add('dark');
}

// Set initial icon based on theme
const icon = themeToggle.querySelector('i');
icon.classList.add(savedTheme === 'dark' ? 'fa-sun' : 'fa-moon');

// Toggle theme on button click
themeToggle.addEventListener('click', () => {
    body.classList.toggle('dark');
    const isDark = body.classList.contains('dark');
    
    // Save theme to localStorage
    localStorage.setItem('theme', isDark ? 'dark' : 'light');

    // Update icon
    icon.classList.toggle('fa-moon');
    icon.classList.toggle('fa-sun');
});

// ---------------------- Language Toggle ----------------------
const langToggle = document.getElementById('langToggle');
let currentLang = localStorage.getItem('language') || 'en';

// Set button text on load
langToggle.textContent = currentLang.toUpperCase();

// Function to update language dynamically
function updateLanguage() {
    document.querySelectorAll('[data-en]').forEach(element => {
        const text = currentLang === 'en' ? element.getAttribute('data-en') : element.getAttribute('data-hi');
        if (text) {
            element.textContent = text;
        }
    });
}

// Apply saved language on page load
updateLanguage();

// Handle language toggle button click
langToggle.addEventListener('click', () => {
    currentLang = currentLang === 'en' ? 'hi' : 'en';
    
    // Save language to localStorage
    localStorage.setItem('language', currentLang);

    // Update button text
    langToggle.textContent = currentLang.toUpperCase();

    // Update language content
    updateLanguage();
});

// ---------------------- Smooth Scroll ----------------------
document.querySelectorAll('nav a').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const section = link.getAttribute('href').replace('.html', '');
        const element = document.getElementById(section);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
        } else {
            // Redirect to the page if section not found
            window.location.href = link.getAttribute('href');
        }
    });
});

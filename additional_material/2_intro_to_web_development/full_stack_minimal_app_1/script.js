// API base URL - change this if backend runs on different port
const API_BASE = 'http://localhost:8000';

// Load messages when page loads
document.addEventListener('DOMContentLoaded', loadMessages);

// Handle form submission
document.getElementById('messageForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const name = document.getElementById('nameInput').value;
    const message = document.getElementById('messageInput').value;
    
    // Create form data for POST request
    const formData = new FormData();
    formData.append('name', name);
    formData.append('message', message);
    
    // Send to backend
    const response = await fetch(`${API_BASE}/api/messages`, {
        method: 'POST',
        body: formData
    });
    
    if (response.ok) {
        // Clear form and reload messages
        document.getElementById('messageForm').reset();
        loadMessages();
    } else {
        alert('Error adding message');
    }
});

// Function to load and display messages
async function loadMessages() {
    try {
        const response = await fetch(`${API_BASE}/api/messages`);
        const data = await response.json();
        
        const messagesDiv = document.getElementById('messages');
        messagesDiv.innerHTML = '';
        
        data.messages.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            messageDiv.innerHTML = `<strong>${msg.name}:</strong> ${msg.message}`;
            messagesDiv.appendChild(messageDiv);
        });
    } catch (error) {
        console.error('Error loading messages:', error);
    }
}
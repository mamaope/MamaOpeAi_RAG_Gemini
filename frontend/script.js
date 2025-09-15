const API_URL = 'http://localhost:8090';
let chatHistory = [];

// Initialize chat
document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    
    // Enter key to send message
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const buttonText = document.getElementById('buttonText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Disable input and show loading
    messageInput.disabled = true;
    sendButton.disabled = true;
    buttonText.style.display = 'none';
    loadingSpinner.style.display = 'inline';
    
    // Add user message to chat
    addMessage('user', message);
    messageInput.value = '';
    
    try {
        // Prepare API request
        const requestBody = {
            patient_data: message,
            chat_history: formatChatHistory()
        };
        
        // Call diagnosis API
        const response = await fetch(`${API_URL}/api/v1/diagnose`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Add AI response to chat
        addMessage('ai', data.model_response, data.diagnosis_complete);
        
        // Update chat history
        chatHistory.push({
            user: message,
            ai: data.model_response
        });
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('error', `Sorry, there was an error processing your request: ${error.message}`);
    } finally {
        // Re-enable input
        messageInput.disabled = false;
        sendButton.disabled = false;
        buttonText.style.display = 'inline';
        loadingSpinner.style.display = 'none';
        messageInput.focus();
    }
}

function addMessage(type, content, diagnosisComplete = false) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    if (type === 'user') {
        messageDiv.innerHTML = `
            <div class="message-content">
                <strong>Patient Case:</strong><br>
                ${formatUserMessage(content)}
            </div>
        `;
    } else if (type === 'ai') {
        messageDiv.innerHTML = `
            <div class="message-content">
                ${formatAIResponse(content)}
            </div>
        `;
    } else if (type === 'error') {
        messageDiv.innerHTML = `
            <div class="message-content">
                <strong>⚠️ Error:</strong><br>
                ${content}
            </div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatUserMessage(message) {
    return message.replace(/\n/g, '<br>');
}

function formatAIResponse(response) {
    // Parse and format the AI response for better readability
    let formattedResponse = response;
    
    // Format sections with proper headers
    formattedResponse = formattedResponse
        .replace(/\*\*Question:\*\*/g, '<h4>🤔 Question:</h4>')
        .replace(/\*\*Rationale:\*\*/g, '<h4>💭 Rationale:</h4>')
        .replace(/\*\*Impression:\*\*/g, '<h4>🔍 Clinical Impression:</h4>')
        .replace(/\*\*Further Management:\*\*/g, '<h4>📋 Further Management:</h4>')
        .replace(/\*\*Sources:\*\*/g, '<h4>📚 Knowledge Base Sources:</h4>')
        .replace(/\*\*ALERT:\*\*/g, '<h4 style="color: #dc3545;">🚨 ALERT:</h4>');
    
    // Format numbered lists
    formattedResponse = formattedResponse.replace(/(\d+\.)\s/g, '<br><strong>$1</strong> ');
    
    // Format bullet points
    formattedResponse = formattedResponse.replace(/^- /gm, '• ');
    
    // Format line breaks
    formattedResponse = formattedResponse.replace(/\n/g, '<br>');
    
    // Highlight the disclaimer
    formattedResponse = formattedResponse.replace(
        /\*(.*application is designed to provide supportive health information.*)\*/,
        '<div class="disclaimer">$1</div>'
    );
    
    // Special formatting for sources section
    if (formattedResponse.includes('📚 Knowledge Base Sources:')) {
        const parts = formattedResponse.split('📚 Knowledge Base Sources:');
        if (parts.length > 1) {
            const sourcesContent = parts[1].split('<div class="disclaimer">')[0].trim();
            const disclaimer = parts[1].includes('<div class="disclaimer">') ? 
                parts[1].split('<div class="disclaimer">')[1] : '';
            
            formattedResponse = parts[0] + 
                '<div class="sources-section"><strong>📚 Knowledge Base Sources:</strong><br>' + 
                sourcesContent + '</div>' +
                (disclaimer ? '<div class="disclaimer">' + disclaimer : '');
        }
    }
    
    return formattedResponse;
}

function formatChatHistory() {
    if (chatHistory.length === 0) return '';
    
    return chatHistory.map(entry => 
        `User: ${entry.user}\nAI: ${entry.ai}`
    ).join('\n\n');
}

// Clear chat function (optional)
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="system-message">
            <p>👋 Chat cleared! Please enter patient information and symptoms for diagnostic assistance.</p>
        </div>
    `;
    chatHistory = [];
} 
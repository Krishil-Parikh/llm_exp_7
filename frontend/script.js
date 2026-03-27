/**
 * QA Agent — Frontend Chat Logic
 * Handles SSE streaming, message rendering, and UI interactions.
 */

// ── State ────────────────────────────────────────
let isProcessing = false;

// ── DOM Elements ─────────────────────────────────
const chatContainer = document.getElementById('chat-container');
const messagesDiv = document.getElementById('messages');
const welcomeScreen = document.getElementById('welcome-screen');
const userInput = document.getElementById('user-input');
const btnSend = document.getElementById('btn-send');
const btnClear = document.getElementById('btn-clear');
const btnTools = document.getElementById('btn-tools');
const btnCloseTools = document.getElementById('btn-close-tools');
const toolsPanel = document.getElementById('tools-panel');
const toolsGrid = document.getElementById('tools-grid');
const modelBadge = document.getElementById('model-badge');

// ── Initialization ───────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadTools();
    setupEventListeners();
});

// ── Health Check ─────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();

        const dot = modelBadge.querySelector('.status-dot');
        if (data.status === 'healthy' && data.model_available) {
            dot.className = 'status-dot connected';
            modelBadge.innerHTML = `<span class="status-dot connected"></span> ${data.model} • Online`;
        } else if (data.status === 'healthy') {
            dot.className = 'status-dot error';
            modelBadge.innerHTML = `<span class="status-dot error"></span> Model not found`;
        } else {
            dot.className = 'status-dot error';
            modelBadge.innerHTML = `<span class="status-dot error"></span> Ollama disconnected`;
        }
    } catch {
        modelBadge.innerHTML = `<span class="status-dot error"></span> Server offline`;
    }
}

// ── Load Tools ───────────────────────────────────
async function loadTools() {
    try {
        const res = await fetch('/api/tools');
        const data = await res.json();

        const icons = {
            calculator: '🔢',
            wikipedia: '📚',
            web_search: '🔍',
            datetime: '📅',
        };

        toolsGrid.innerHTML = data.tools.map(tool => `
            <div class="tool-card">
                <div class="tool-card-name">${icons[tool.name] || '🔧'} ${tool.name}</div>
                <div class="tool-card-desc">${tool.description}</div>
            </div>
        `).join('');
    } catch {
        toolsGrid.innerHTML = '<p style="color: var(--text-muted);">Unable to load tools.</p>';
    }
}

// ── Event Listeners ──────────────────────────────
function setupEventListeners() {
    // Send message
    btnSend.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });

    // Clear history
    btnClear.addEventListener('click', clearHistory);

    // Tools panel toggle
    btnTools.addEventListener('click', () => {
        toolsPanel.classList.toggle('visible');
    });
    btnCloseTools.addEventListener('click', () => {
        toolsPanel.classList.remove('visible');
    });

    // Suggestion chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            userInput.value = chip.dataset.query;
            sendMessage();
        });
    });
}

// ── Send Message ─────────────────────────────────
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isProcessing) return;

    isProcessing = true;
    btnSend.disabled = true;

    // Hide welcome screen
    welcomeScreen.classList.add('hidden');

    // Add user message to UI
    appendMessage('user', message);

    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';

    // Show thinking indicator
    const thinkingEl = showThinking();

    // Create assistant message container
    const { container: assistantContainer, reasoningChain, bubbleEl } = createAssistantMessage();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        // Remove thinking indicator
        thinkingEl.remove();

        // Read SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events
            const events = buffer.split('\n\n');
            buffer = events.pop(); // Keep incomplete event in buffer

            for (const event of events) {
                if (!event.startsWith('data: ')) continue;
                const jsonStr = event.slice(6);

                try {
                    const step = JSON.parse(jsonStr);
                    handleAgentStep(step, reasoningChain, bubbleEl);
                } catch (parseErr) {
                    console.error('Parse error:', parseErr);
                }
            }
        }
    } catch (err) {
        thinkingEl.remove();
        bubbleEl.innerHTML = `<p style="color: var(--error);">Error: ${err.message}. Make sure the server is running.</p>`;
    }

    isProcessing = false;
    btnSend.disabled = false;
    userInput.focus();
    scrollToBottom();
}

// ── Handle Agent Step ────────────────────────────
function handleAgentStep(step, reasoningChain, bubbleEl) {
    const icons = {
        thought: '💭',
        action: '⚡',
        action_input: '📥',
        observation: '👁️',
        error: '❌',
    };

    const labels = {
        thought: 'Thought',
        action: 'Action',
        action_input: 'Input',
        observation: 'Result',
        error: 'Error',
    };

    if (step.type === 'final_answer') {
        bubbleEl.innerHTML = renderMarkdown(step.content);
    } else if (step.type === 'done') {
        // Stream finished
        return;
    } else {
        // Add reasoning step
        const stepEl = document.createElement('div');
        stepEl.className = `reasoning-step ${step.type}`;
        stepEl.innerHTML = `
            <span class="step-icon">${icons[step.type] || '•'}</span>
            <span class="step-label">${labels[step.type] || step.type}:</span>
            <span class="step-content">${escapeHtml(step.content)}</span>
        `;
        reasoningChain.appendChild(stepEl);
    }

    scrollToBottom();
}

// ── UI Helpers ───────────────────────────────────
function appendMessage(role, content) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;

    const avatar = role === 'user' ? 'U' : 'AI';

    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">${role === 'user' ? escapeHtml(content) : renderMarkdown(content)}</div>
        </div>
    `;

    messagesDiv.appendChild(messageEl);
    scrollToBottom();
}

function createAssistantMessage() {
    const container = document.createElement('div');
    container.className = 'message assistant';

    const reasoningChain = document.createElement('div');
    reasoningChain.className = 'reasoning-chain';

    const bubbleEl = document.createElement('div');
    bubbleEl.className = 'message-bubble';

    container.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content"></div>
    `;

    const contentDiv = container.querySelector('.message-content');
    contentDiv.appendChild(reasoningChain);
    contentDiv.appendChild(bubbleEl);

    messagesDiv.appendChild(container);

    return { container, reasoningChain, bubbleEl };
}

function showThinking() {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="thinking-indicator">
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
                <span class="thinking-text">Reasoning...</span>
            </div>
        </div>
    `;
    messagesDiv.appendChild(el);
    scrollToBottom();
    return el;
}

async function clearHistory() {
    try {
        await fetch('/api/history', { method: 'DELETE' });
        messagesDiv.innerHTML = '';
        welcomeScreen.classList.remove('hidden');
    } catch (err) {
        console.error('Failed to clear history:', err);
    }
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ── Markdown Rendering (Simple) ──────────────────
function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" style="color: var(--accent-primary);">$1</a>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

let conversationHistory = [];
        let isLoading = false;

        function scrollToBottom() {
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function addMessage(content, isUser, isLoading = false) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            if (isLoading) {
                contentDiv.innerHTML = `
                    <div class="loading">
                        <span>思考中</span>
                        <div class="loading-dots">
                            <div class="loading-dot"></div>
                            <div class="loading-dot"></div>
                            <div class="loading-dot"></div>
                        </div>
                    </div>
                `;
                contentDiv.id = 'loading-message';
            } else {
                contentDiv.textContent = content;
            }
            
            messageDiv.appendChild(contentDiv);
            chatMessages.appendChild(messageDiv);
            scrollToBottom();
            
            return messageDiv;
        }

        function handleEnter(event) {
            if (event.key === 'Enter' && !isLoading) {
                sendMessage();
            }
        }

        async function sendMessage() {
            if (isLoading) return;
            
            const messageInput = document.getElementById('messageInput');
            const sendBtn = document.getElementById('sendBtn');
            const message = messageInput.value.trim();
            
            if (!message) return;
            
            // 禁用輸入
            isLoading = true;
            messageInput.disabled = true;
            sendBtn.disabled = true;
            sendBtn.textContent = '發送中...';
            
            // 添加用戶訊息
            addMessage(message, true);
            conversationHistory.push({role: 'user', content: message});
            
            // 添加載入訊息
            const loadingMessage = addMessage('', false, true);
            
            messageInput.value = '';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        history: conversationHistory
                    })
                });
                
                const data = await response.json();
                
                // 移除載入訊息
                loadingMessage.remove();
                
                if (data.error) {
                    addMessage(`錯誤：${data.error}`, false);
                } else {
                    addMessage(data.response, false);
                    conversationHistory.push({role: 'model', content: data.response});
                }
                
            } catch (error) {
                // 移除載入訊息
                loadingMessage.remove();
                addMessage(`連接錯誤：${error.message}`, false);
            }
            
            // 重新啟用輸入
            isLoading = false;
            messageInput.disabled = false;
            sendBtn.disabled = false;
            sendBtn.textContent = '發送';
            messageInput.focus();
        }

        // 頁面載入完成後聚焦輸入框
        window.onload = function() {
            document.getElementById('messageInput').focus();
        }
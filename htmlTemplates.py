css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTRzmXD1-lrVoIiWqnJgNwSau8WM5AQ_m1FjQ&s" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/jpeg;base64,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">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
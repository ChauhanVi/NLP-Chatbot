# ⚙️ Setup Instructions

 1️⃣ Clone Repository
git clone https://github.com/ChauhanVi/NLP-chatbot.git

cd NLP-chatbot

2️⃣ Create Virtual Environment (Recommended)

python -m venv .venv

 Activate venv
 
.\.venv\Scripts\activate     # Windows

source .venv/bin/activate  # Linux/Mac

 3️⃣ Train the Model
python train_chatbot.py

This generates:

chatbot_model.h5 (trained model)

classes.pkl (intents/classes)

words.pkl (tokenized words)

▶️ How to Use

Run the Chatbot (Console)

python chat.py

Opens a local web server at: http://127.0.0.1:5000

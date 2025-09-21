from flask import Flask, render_template, request, jsonify
import chat

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    ints = chat.predict_class(msg)
    res = chat.get_response(ints, chat.intents)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(debug=True)

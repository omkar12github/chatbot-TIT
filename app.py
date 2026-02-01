from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQ CSV
data = pd.read_csv("tit_faq.csv")
questions = data["question"].tolist() 
answers = data["answer"].tolist()

# Convert questions into vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    sim = cosine_similarity(user_vec, tfidf_matrix)
    idx = np.argmax(sim)

    # If similarity is low → no matching answer
    if sim[0][idx] < 0.20:
        return "Sorry, I don't have an answer for that."

    return answers[idx]

@app.route("/", methods=["GET", "POST"])
def home():
    user_msg = ""          # ✅ ADD
    bot_reply = ""

    if request.method == "POST":
        user_msg = request.form.get("message")     # ✅ KEEP user message
        bot_reply = chatbot_response(user_msg)

    return render_template(
        "index.html",
        reply=bot_reply,
        user_msg=user_msg      # ✅ SEND user message to HTML
    )


if __name__ == "__main__":
    app.run(debug=True)

































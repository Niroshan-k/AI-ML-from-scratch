from flask import Flask, request, jsonify
import openai
from openai import OpenAI
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def search_brave(query):
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": os.getenv("BRAVE_API_KEY")
    }
    params = {
        "q": query
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    # Get the top 3-5 results' snippets
    results = data.get("web", {}).get("results", [])
    snippets = [item.get("description", "") for item in results[:5]]
    return " ".join(snippets)

def summarize_text(text, question):
    prompt = f"Summarize the following to answer the question: '{question}'\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content



@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    info = info = search_brave(question)
    answer = summarize_text(info, question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)

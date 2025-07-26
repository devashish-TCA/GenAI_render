from flask import Flask, request, jsonify
import os
from GenAI_Chatbot import ask_gemini

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    answer, sources = ask_gemini(query)
    return jsonify({'answer': answer, 'sources': sources})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 
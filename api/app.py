import json
import requests
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


@app.route('/api/get_next_word', methods=['POST'])
def get_next_word():
    index2word = op_js("api/index2word.json")
    headers = {"content-type": "application/json"}
    res = requests.post('http://3.235.192.240:8501/v1/models/next-bacon-prediction:predict',
                        data=request.json, headers=headers)
    prediction = json.loads(res.text)['predictions'][0][:-1]
    return index2word[np.argmax(prediction)]


def op_js(filename):
    with open(filename) as f_in:
        return json.load(f_in)
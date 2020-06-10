import json
import requests
import numpy as np
from flask import Flask, request

app = Flask(__name__)


@app.route('/api/get_next_word', methods=['POST'])
def get_next_word():
    index2word = {int(k):v for k,v in op_js("api/index2word.json").items()}
    headers = {"content-type": "application/json"}
    res = requests.post('http://3.235.192.240:8501/v1/models/next-bacon-prediction:predict',
                        data='{"instances": ["phrase"]}'.replace("phrase", str(request.data.decode("utf-8"))),
                        headers=headers)
    prediction = json.loads(res.text)['predictions'][0][:-1]
    return {"next_word": index2word[np.argmax(prediction)]}


# method to read the index to word mapping
def op_js(filename):
    with open(filename) as f_in:
        return json.load(f_in)


if __name__ == "__main__":
    app.run()

# next-bacon-prediction

This demo of uses the Universal Sentence Encoder to build a next-word prediction service. Francis Bacon was an English
essayist famous for his aphoristic style. A disregard for traditional essay convention noticeably marks his work.
The model in this demo is trained on his essay _Of Truth_.


# Process

The code in this repo is meant as a demo for how a similar service can be built. The model has not be tuned or evaluated.
Similarly, there is no unit tests or deployment automation mechanisms. I'd be happy to provide more detail on either. Please reach out!

### Training
The model was built in a cloud notebook using the 1999 most frequently occuring words. To construct the train set, the 
corpus was windowed over and randomly, phrases of
of 5-15 words were samples. The label is simply the next word in each phrase. 

The first layer of the model is the Universal Sentence Encoder which is hosted on TensorFlow Hub. We specifically use version
5 because previous ones have comparability issues with Keras' training and inference methods. It is also good to note that
the embeding should be read as a `hub.KerasLayer`. Importing it as a regular tf module and building a custom Keras layer with 
a `Lambda` function can result in serialization errors, regardless of save API, when saving the model. tf and Keras operations 
still don't always play nice 😔. 

The output of the model is a 2000 (1999+UKN) element matrix of next-word probabilities.

After the model was trained, the Keras save API was used to export the model as a Protobuf file and index files. 
The dictionary containing `index2word` mappings was exported as a JSON file. 

### Deployment

The model is hosted on an EC2 (at 3.235.192.240:8501) instance running the [TensorFlow Serving image](https://www.tensorflow.org/tfx/serving/docker).
tf Serving requires models to be versioned by directory. Here, `v1` is the where our model lives. Within this directory, 
the following artifacts, which are generated through the Keras save API, are added.

```
|__ v1
    |__ saved_model.pb
    |__ assets (optional)
    |__ variables
        |__ variables.index
        |__ variables.data-00000-of-00001
```

Our `variables.data-00000-of-00001` is around 600MB so it has been omitted from GitHub. 

To request a prediction(s):
```shell script
$ curl -d '{"instances": [<phrase1>, <phrase2>, ...]}' \ 
  -X POST http:/3.235.192.240:8501/v1/models/next-bacon-prediction:predict

# Returns => { "preidctions:" [0.2, 0.4, ...], [0.5,0.43, ...]}
```

For our example we would only like the next word, not the entire probability matrix. To do this we can set up a Flask API 
which contains the `index2word` mapping (see `/api` directory). I have provided an example Dockerfile to run it. It will ping the tf Serving
endpoint then use `index2word` to convert the matrix to
the most likely next word. Having a second service enables us to customize what we would like to return. The service can 
be easily extended to return the top 5 most likely next words. Unlike the tf serving API, it only supports 1 prediction per
request.

ex: 
```python
import requests
requests.post("http://3.235.192.240:5000/api/get_next_word",
                        data="the world")
# Returns => {"next_word":"governs"}
```

Thanks for reading!

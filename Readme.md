## Image reconstruction Web API Sample (keras)

### What is this?
This is a sample of Image reconstruction web API using autoencoder.
In this sample, we reconstruct input image based on CIFAR10 trained model.

You can replace CIFAR10 model with your trained model and modify some codes and test your model easily.


#### Implemented API

- GET /predict.json	 
	- required params
		- url or data (base64 encoded)
- POST /predict.json
	- required params
		- file

#### API Test Pages

- /upload

#### Test on local 

```
$ python main.py

```

#### Deploy to heroku

```
$ git clone git@github.com:tanakatsu/keras-autoencoder-api-sample.git
$ heroku create
$ git push heroku master
$ heroku open
```


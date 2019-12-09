import os
from flask import Flask, render_template, request, flash
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from cv2 import imread
from skimage.transform import resize

app = Flask(__name__)
app.secret_key = "super secret key"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model1=None

graph=None


target = os.path.join(APP_ROOT, 'static/')


def my_model():
	global model1
	model1=load_model("/".join([APP_ROOT,'model.h5']))
	global graph
	graph = tf.get_default_graph()

def img(destination):
	img_file = imread(destination)
	img_file = resize(img_file, (150, 150, 3))
	img_arr = np.asarray(img_file)
	img_arr=np.expand_dims(img_arr,axis=0)
	return img_arr


@app.route('/')
def index():
	return render_template('index.html')

@app.route("/upload", methods=['POST'])
def upload():
	if request.method=="POST":
		file = request.files["skin"]
		filename = file.filename
		destination = "/".join([target, filename])
		file.save(destination)
		img_arr=img(destination)
		global graph
		prob1=None
		with graph.as_default():
			prob1=model1.predict(img_arr)[0][0] * 100
			prob1=round(prob1,2)
		res1="The chances of melanoma are " + str(prob1)+ " %."
		flash(res1)
		os.remove(destination) 
		#K.clear_session()
		return render_template('index.html')


if __name__ == "__main__":
	my_model()
	app.run(port=5000, debug=True)
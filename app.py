from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import scale

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/ress',methods = ['POST'])
def index2():
	res="No"
	li=[]
	for i in range(1,179):
		s="X"+str(i)
		inp=request.form[s]
		li.append(int(inp))
	X_test=li
	X_test=scale(X_test)
	X_test=list(X_test)
	X_test=[X_test]
	X_test=np.asarray(X_test)
	model = pickle.load(open("finalized_model.sav",'rb'))
	y_pred = model.predict(X_test)
	prediction = np.round(y_pred)[0]
	res="Yes" if (prediction==1) else "No"
	return render_template('index.html',variable=res)

if __name__ == "__main__":
	app.run(debug=True) 
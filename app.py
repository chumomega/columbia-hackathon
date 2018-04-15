from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/hello')
def hello():
	return render_template('hello.html')

@app.route('/isOdd/<int:num>')
def isOdd(num):
	if num%2==1:
		return str(num) + ' is odd'
	else:
		return str(num) + ' is even'

@app.route('/records')
def records():
	med_record = {'name': 'Chuma', 'age':20, 'blood_type':'O'}
	return render_template('record.html', record=med_record)
#html-variable = python-variable


@app.route('/people')
def people():
    personsList = ['chuma', 'phillip', 'chika']
    return render_template('list.html', persons=personsList)
    
if __name__ =='__main__':
	app.run(debug=True, use_reloader=True)
    





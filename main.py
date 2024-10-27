# Model training
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")
data.head()

x = data.iloc[:,2:7].values

x.shape

y = data.iloc[:,-1].values

y.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

k = int(np.sqrt(x_train.shape[0]))

knn = KNeighborsClassifier(n_neighbors=k)

#train
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

y_pred.shape

accuracy_score(y_test, y_pred)
print(accuracy_score)

user_input = [[4,1,4,7,4],[4,1,4,7,4]]
output = knn.predict(user_input)
print(output[0])

# Implementing the model with a UI
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='template', static_folder='static')

# Normal Redirects
@app.route('/')
def index():
    return render_template('PCS.html')

@app.route('/Openness')
def O():
    return render_template('openness.html')

@app.route('/Conscientiousness')
def C():
    return render_template('conscientiousness.html')

@app.route('/Extraversion')
def E():
    return render_template('extraversion.html')

@app.route('/Agreeableness')
def A():
    return render_template('agreeableness.html')

@app.route('/Neuroticism')
def N():
    return render_template('neuroticism.html')

@app.route('/Personality Classification')
def PCS():
    return render_template('index.html')

#Functional Redirects
@app.route('/Results1', methods=['POST'])
def show1():
    o1 = int(request.form['o1'])
    o2 = int(request.form['o2'])
    o3 = int(request.form['o3'])
    o4 = int(request.form['o4'])
    o5 = int(request.form['o5'])
    op = int((o1+o2+o3+o4+o5)/5)
    op = str(op)+" in Openness"
    print(op)
    return render_template('result.html', op = op)

@app.route('/Results2', methods=['POST'])
def show2():
    c1 = int(request.form['c1'])
    c2 = int(request.form['c2'])
    c3 = int(request.form['c3'])
    c4 = int(request.form['c4'])
    c5 = int(request.form['c5'])
    op = int((c1+c2+c3+c4+c5)/5)
    op = str(op)+" in Conscientiousness"
    print(op)
    return render_template('result.html', op = op)

@app.route('/Results3', methods=['POST'])
def show3():
    e1 = int(request.form['e1'])
    e2 = int(request.form['e2'])
    e3 = int(request.form['e3'])
    e4 = int(request.form['e4'])
    e5 = int(request.form['e5'])
    op = int((e1+e2+e3+e4+e5)/5)
    op = str(op)+" in Extraversion"
    print(op)
    return render_template('result.html', op = op)

@app.route('/Results4', methods=['POST'])
def show4():
    a1 = int(request.form['a1'])
    a2 = int(request.form['a2'])
    a3 = int(request.form['a3'])
    a4 = int(request.form['a4'])
    a5 = int(request.form['a5'])
    op = int((a1+a2+a3+a4+a5)/5)
    op = str(op)+" in Agreeableness"
    print(op)
    return render_template('result.html', op = op)

@app.route('/Results5', methods=['POST'])
def show5():
    n1 = int(request.form['n1'])
    n2 = int(request.form['n2'])
    n3 = int(request.form['n3'])
    n4 = int(request.form['n4'])
    n5 = int(request.form['n5'])
    op = int((n1+n2+n3+n4+n5)/5)
    op = str(op)+" in Neuroticism"
    print(op)
    return render_template('result.html', op = op)

@app.route('/Results6', methods=['POST'])
def show6():
    openness = int(request.form['openness'])
    neuroticism = int(request.form['neuroticism'])
    conscientiousness = int(request.form['conscientiousness'])
    agreeableness = int(request.form['agreeableness'])
    extraversion = int(request.form['extraversion'])
    inputValue = [[openness, neuroticism, conscientiousness, agreeableness, extraversion]]
    op = knn.predict(inputValue)
    print(op)
    if(op == ['serious']):
        op = "serious"
    elif(op == ['dependable']):
        op = "dependable"
    elif(op == ['extraverted']):
        op = "extraverted"
    elif(op == ['lively']):
        op = "lively"
    elif(op == ['responsible']):
        op = "responsible"
    return render_template('output.html', op = op)

if __name__ == '__main__':
    app.run(debug=True)
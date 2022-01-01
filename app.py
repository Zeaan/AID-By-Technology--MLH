from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/help')
def help():
    return render_template("help.html")

@app.route('/breastCancer', methods=['GET','POST'])
def breastCancer():
    pickle_in = open("Breast_cancer.pickle","rb")
    model = pickle.load(pickle_in)
    if request.method=="POST":
        mean_radius = float(request.form["mean_radius"])
        mean_texture = float(request.form["mean_texture"])
        mean_perimeter = float(request.form["mean_perimeter"])
        mean_area = float(request.form["mean_area"])
        mean_smoothness = float(request.form["mean_smoothness"])
        diagnosis = model.predict( [[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]] ).tolist()
        prediction = diagnosis[0]
        return render_template("result.html", prediction=prediction, disease="Breast Cancer")
    return render_template("breastCancer.html")

@app.route('/diabetes', methods=['GET','POST'])
def diabetes():
    pickle_in = open("Diabetes.pickle","rb")
    model = pickle.load(pickle_in)
    if request.method=="POST":
        Pregnancies = float(request.form["Pregnancies"])
        Glucose = float(request.form["Glucose"])
        BloodPressure = float(request.form["BloodPressure"])
        SkinThickness = float(request.form["SkinThickness"])
        Insulin = float(request.form["Insulin"])
        BMI = float(request.form["BMI"])
        Age = float(request.form["Age"])
        Outcome = model.predict( [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,Age]] ).tolist()
        prediction = Outcome[0]
        return render_template("result.html", prediction=prediction, disease="Diabetes")
    return render_template("diabetes.html")

if __name__=='__main__':
    app.run(debug=True, port=8000)

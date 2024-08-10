from flask import Flask, render_template, request
from crop import prediction
app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    # get the input values from the form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    pH = float(request.form['pH'])
    EC = float(request.form['EC'])
    OC = float(request.form['OC'])
    B = float(request.form['B'])
    Zn = float(request.form['Zn'])
    Fe = float(request.form['Fe'])
    Mn = float(request.form['Mn'])
    Cu = float(request.form['Cu'])
    S = float(request.form['S'])

    # make a prediction using the crop prediction model
    predicted_crop = prediction(N, P, K, pH, EC, OC, B, Zn, Fe, Mn, Cu, S)

    # render the prediction result template with the predicted crop
    return render_template('result.html', predicted_crop=predicted_crop)

if __name__ == '__main__':
    app.run(debug=True)
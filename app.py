from flask import Flask, request, render_template
# import pickle, sklearn
import joblib
import pandas as pd
model = joblib.load('model.pkl')


app = Flask(__name__)

@app.route('/')

def home():


    return render_template('index.html')
@app.route('/predict', methods = ['POST'])
def predict():
    d = {}
    i = 0
    val = [int (x) for x in request.form.values()]
    for col in ['housing_median_age','total_rooms','total_bedrooms','population',	'households',	'median_income',	'ocean_proximity']:
        d[col] = val[i]
        i += 1
    inputs = pd.DataFrame(d, index=[0])
    bd = pd.read_json('donnees_entrees_form.json')
    pd.concat([bd, inputs],0).reset_index(drop=True).to_json("donnees_entrees_form.json")

    #inputs.to_csv('donnees_entrees_formulaire.csv')
    resultat = model.predict(inputs)
    return render_template('index.html', prediction_text='Resultat prédiction prix: {}'.format(resultat))



if __name__ == "__main__":
    app.run(debug=True)
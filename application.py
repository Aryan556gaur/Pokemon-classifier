from flask import Flask,request,render_template,jsonify
from src.pipelines.PredictionPipeline import Prediction_pipeline,CustomData


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Type_1 = request.form.get('Type_1'),
            Total=float(request.form.get('Total')),
            HP = float(request.form.get('HP')),
            Attack = float(request.form.get('Attack')),
            Defense = float(request.form.get('Defense')),
            Sp_Atk = float(request.form.get('Sp_Atk')),
            Sp_Def = float(request.form.get('Sp_Def')),
            Speed = float(request.form.get('Speed')),
            Generation = float(request.form.get('Generation')),
            Color= request.form.get('Color'),
            hasGender = request.form.get('hasGender'),
            Pr_Male = float(request.form.get('Pr_Male')),
            Egg_Group_1 = request.form.get('Egg_Group_1'),
            hasMegaEvolution = request.form.get('hasMegaEvolution'),
            Height_m = float(request.form.get('Height_m')),
            Weight_kg = float(request.form.get('Weight_kg')),
            Catch_Rate = float(request.form.get('Catch_Rate')),
            Body_Style = request.form.get('Body_Style')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=Prediction_pipeline()
        pred=predict_pipeline.initiate_prediction(final_new_data)

        results=pred[0]

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
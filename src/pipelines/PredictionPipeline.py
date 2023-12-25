import os,sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
from dataclasses import dataclass

@dataclass
class Prediction_pipeline_config():
    prediction_output_path = os.path.join("predictions","predictions.csv")

class Prediction_pipeline:

    def __init__(self):
        pass

    def initiate_prediction(self,features):
        preprocessor_path = os.path.join("artifacts", "Preprocessor.pkl")
        model_path = os.path.join("artifacts", "model.pkl")

        try:
            preprocessor_obj = load_object(file_path=preprocessor_path)
            model_obj = load_object(file_path=model_path)

            x = preprocessor_obj.transform(features)
            y = model_obj.predict(x)

            return y

        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, Type_1, Total, HP, Attack, Defense, Sp_Atk, Sp_Def, Speed, Generation, Color, hasGender, 
                 Pr_Male, Egg_Group_1,hasMegaEvolution, Height_m, Weight_kg, Catch_Rate,Body_Style):
        self.Type_1 = Type_1
        self.Total = Total
        self.HP = HP
        self.Attack = Attack
        self.Defense = Defense
        self.Sp_Atk = Sp_Atk
        self.Sp_Def = Sp_Def
        self.Speed = Speed
        self.Generation = Generation
        self.Color = Color
        self.hasGender = hasGender
        self.Pr_Male = Pr_Male
        self.Egg_Group_1 = Egg_Group_1
        self.hasMegaEvolution = hasMegaEvolution
        self.Height_m = Height_m
        self.Weight_kg = Weight_kg
        self.Catch_Rate = Catch_Rate
        self.Body_Style = Body_Style

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {"Type_1":[self.Type_1], "Total":[self.Total], "HP":[self.HP], 
                "Attack":[self.Attack], "Defense":[self.Defense], "Sp_Atk":[self.Sp_Atk], "Sp_Def":[self.Sp_Def],
                "Speed":[self.Speed], "Generation":[self.Generation], "Color":[self.Color], "hasGender":[self.hasGender], 
                "Pr_Male":[self.Pr_Male], "Egg_Group_1":[self.Egg_Group_1],"hasMegaEvolution":[self.hasMegaEvolution],
                "Height_m":[self.Height_m], "Weight_kg":[self.Weight_kg], "Catch_Rate":[self.Catch_Rate],"Body_Style":[self.Body_Style]}
            
            df = pd.DataFrame(custom_data_input_dict)

            logging.info("dataframe created")

            return df
        
        except Exception as e:
            logging.info("Exception occured in prediction pipeline")
            raise CustomException(e,sys)
        
class Batch_prediction:
    def __init__(self, request):
        self.request = request
        self.prediction_config = Prediction_pipeline_config()

    def save_data(self):
        input_file = self.request.files["file"]
        input_file_path = os.makedirs(os.path.join("input_data",input_file.filename),exist_ok=True)
        input_file.save(input_file_path)

        return input_file_path
    
    def initiate_file_prediction(self,input_file_path):
        df = pd.read_csv(input_file_path)

        for col in df.columns:
            if col in ["Number","Egg_group_2","Type_2"]:
                df.drop(col,axis=1,inplace=True)

        preprocessor_path = os.path.join("artifacts", "Preprocessor.pkl")
        model_path = os.path.join("artifacts", "model.pkl")

        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)

        x = preprocessor.transform(df)
        y = model.predict(x)

        df["Target"] = [i for i in y]
        df["Target"] = df["Target"].map({1:"yes",0:"no"})

        predicted_file_path = os.makedirs(self.prediction_config.prediction_output_path,exist_ok=True)
        df.to_csv(predicted_file_path)

        return predicted_file_path




import os,sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    Preprocessor_path = os.path.join("artifacts", "Preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor_obj(self):
        
        logging.info("Data Preprocessing Starts")

        try:
            num_cols = ['Total','HP','Attack','Defense','Sp_Atk','Sp_Def','Speed','Generation','Pr_Male','Height_m','Weight_kg','Catch_Rate']
            cat_cols = ['Type_1', 'Color', 'hasGender', 'Egg_Group_1', 'hasMegaEvolution','Body_Style']


            cat_type = ['Grass_Bug', 'Fire', 'Water_Ice', 'Normal', 'Poison', 'Electric','Ground_Rock', 'Supernatural', 'Fighting', 'Dragon', 'Steel','Flying']
            cat_col = ['Green', 'Red', 'Blue', 'White', 'Brown', 'Yellow', 'Purple','Pink', 'Grey', 'Black']
            cat_has_gender = [ True, False]
            cat_egg1 = ['Monster', 'Grass_Bug', 'Flying', 'Field', 'Undiscovered', 'Fairy','Water', 'Human-Like', 'Mineral', 'Amorphous', 'Ditto', 'Dragon']
            cat_mega = [False,  True]
            cat_body_style = ['quadruped', 'bipedal_tailed', 'insectoid', 'serpentine_body','four_wings', 'two_wings', 'bipedal_tailless', 'head_legs', 'head_base', 'multiple_bodies', 'several_limbs', 'head_arms','with_fins', 'head_only']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(categories=[cat_type,cat_col,cat_has_gender,cat_egg1,cat_mega,cat_body_style])),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols)
            ])

            logging.info("Preprocessor object obtained")

            return preprocessor
        
        except Exception as e:
            logging.info('Exception occured during Data Preprocessing')
            raise CustomException(e,sys)


    def initiate_DataTransformation(self,train_data,test_data):
        
        try:
            train_data = pd.read_csv(train_data)
            test_data = pd.read_csv(test_data)

            x_train = train_data.drop("isLegendary", axis=1)
            y_train = train_data["isLegendary"]
            x_test = test_data.drop("isLegendary", axis=1)
            y_test = test_data["isLegendary"]

            for i in [x_train,x_test]:

                i["Type_1"] = i["Type_1"].replace(["Grass","Bug"], "Grass_Bug")
                i["Type_1"] = i["Type_1"].replace(["Water","Ice"], "Water_Ice")
                i["Type_1"] = i["Type_1"].replace(["Ground","Rock"], "Ground_Rock")
                i["Type_1"] = i["Type_1"].replace(["Psychic","Ghost", "Dark", "Fairy"], "Supernatural")

                i["Egg_Group_1"] = i["Egg_Group_1"].replace(["Bug","Grass"], "Grass_Bug")
                i["Egg_Group_1"] = i["Egg_Group_1"].replace(["Water_1","Water_2","Water_3"], "Water")

            y_train.map(lambda x: 1 if x else 0)
            y_test.map(lambda x: 1 if x else 0)

            preprocessor_obj = self.get_preprocessor_obj()
            x_train_transformed = preprocessor_obj.fit_transform(x_train)
            x_test_transformed = preprocessor_obj.transform(x_test)

            smt= SMOTETomek(sampling_strategy="minority")
            train_x, train_y = smt.fit_resample(x_train_transformed,y_train)
            test_x, test_y = smt.fit_resample(x_test_transformed,y_test)


            save_object(self.transformation_config.Preprocessor_path, preprocessor_obj)

            logging.info("Data transformaion Completed")

            return train_x,test_x,train_y,test_y,self.transformation_config.Preprocessor_path

        except Exception as e:
            logging.info('Exception occured during Data Transformation')
            raise CustomException(e,sys)







import pickle
import numpy as np
import csv
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load models 
with open('./pkl_files/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('./pkl_files/nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('./pkl_files/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('./pkl_files/data_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f)

with open('./pkl_files/Doctor_Specialist_Model.pkl', 'rb') as f:
    specialization = pickle.load(f)

# prediction function
def predictDisease(symptoms):
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom.capitalize(), -1)
        if index != -1:
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)
    
    rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction], axis=0, keepdims=True, nan_policy='omit')[0][0]

    predictions = {
        "rf_model_prediction": rf_prediction,
        "nb_model_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

# Disease Description function
def diseaseDescription(predicted_disease):
    disease_descriptions={}
    predicted_disease = predicted_disease

    with open('../Data/symptom_Description.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row 

        for row in reader:
            disease_name, description = row  
            disease_descriptions[disease_name] = description

    if predicted_disease in disease_descriptions:
        description = disease_descriptions[predicted_disease]
        print(f"Description of {predicted_disease}: {description}")
    else:
        print(f"Description for {predicted_disease} not found.")

def diseasePrediction(predicted_disease):
    disease_precautions={}
    with open('../Data/symptom_precaution.csv','r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            disease,precaution_1, precaution_2, precaution_3, precaution_4 = row
            disease_precautions[disease]= [precaution_1, precaution_2, precaution_3, precaution_4]
        # print(disease_precautions)

        if predicted_disease in disease_precautions.keys():
            precautions = disease_precautions[predicted_disease]
            print(f"Precautions for {predicted_disease}:")
            for i, precaution in enumerate(precautions):
                # print(i, precaution)
                if precaution == '':
                    continue
                print(f"{precaution.capitalize()}")
        else:
            print(f"Precautions for {predicted_disease} not found.")

def recommend(predicted_disease): 
    if predicted_disease in specialization:
        print (f"For {predicted_disease}, recommend consulting a {specialization[predicted_disease]}.")
    else:
        print (f"No specific recommendation found for {predicted_disease}.")

#TUser input ----> symptoms
Input_Symptoms = input('Enter the Symptoms (comma-separated): ')
disease = predictDisease(Input_Symptoms.split(','))

# print(data_dict["symptom_index"])
predicted_disease = disease["final_prediction"]
print(f'Predicted Disease: {predicted_disease}')
print(f'Your symptoms: {Input_Symptoms}')
diseaseDescription(predicted_disease)
diseasePrediction(predicted_disease)
recommend(predicted_disease)

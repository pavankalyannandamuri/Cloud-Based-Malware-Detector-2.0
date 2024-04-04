import joblib
import os
import json
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
import boto3


"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError("This model only supports application/json input")

"""
predict_fn
    input_data: returned data from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    # Process the input data if necessary
    processed_data = process_input(input_data)
    # Make predictions using the model
    predictions = model.predict(processed_data)
    #print(predictions)
    return predictions

def process_input(input_data):
    # Process input data as needed before passing to the model for prediction
    NgramFeaturesList_pred = np.array(input_data['NgramFeaturesList_pred'])
    importsCorpus_pred = input_data['importsCorpus_pred']
    sectionNames_pred = input_data['sectionNames_pred']
    numSections_pred = int(input_data['numSections_pred'])
    

    # Load featurizers
    imports_featurizer = joblib.load(os.path.join("opt/ml/model", "imports_featurizer.pkl"))
    section_names_featurizer = joblib.load(os.path.join("opt/ml/model", "section_names_featurizer.pkl"))
    #print(NgramFeaturesList_pred, importsCorpus_pred, sectionNames_pred, numSections_pred)
    #print(imports_featurizer, section_names_featurizer)
    # Transform text features
    importsCorpus_pred_transformed = imports_featurizer.transform([importsCorpus_pred])
    sectionNames_pred_transformed = section_names_featurizer.transform([sectionNames_pred])

    # Concatenate features into a single sparse matrix
    processed_data = hstack([csr_matrix(NgramFeaturesList_pred),
                             importsCorpus_pred_transformed,
                             sectionNames_pred_transformed,
                             csr_matrix([numSections_pred]).transpose()])
    #print(processed_data)
    return processed_data

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
"""
def output_fn(prediction, content_type):
    res = int(prediction[0])
    respJSON = {'Output': res}
    return respJSON

"""if __name__ == '__main__':
    predict_fn({'NgramFeaturesList_pred': [[24183, 3382, 304, 17, 923, 636, 358, 275, 128, 635, 358, 613, 389, 384, 448, 12, 380, 170, 307, 122, 224, 203, 51, 338, 521, 111, 395, 215, 175, 419, 264, 397, 287, 106, 487, 236, 16, 277, 459, 594, 469, 241, 155, 163, 158, 230, 215, 443, 80, 46, 44, 216, 68, 42, 36, 48, 161, 29, 240, 145, 139, 52, 20, 75, 99, 33, 224, 161, 38, 226, 729, 139, 27, 168, 19, 68, 269, 271, 236, 33, 197, 207, 337, 1114, 126, 111, 255, 175, 47, 46, 60, 318, 129, 79, 16, 223, 162, 79, 15, 157]],
 'importsCorpus_pred': "kernel32 shlwapi ole32 shell32 user32",
 'sectionNames_pred': ".text .rdata .data .rsrc .reloc",
 'numSections_pred': "5"}, model_fn(""))
"""

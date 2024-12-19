from utils.common import *
import pandas as pd

class ClassificationModel:
    def __init__(self, 
                 job_classifier_model=None, general_question_classifier_model=None, 
                 busy_classifier_model=None, additional_feature_classifier_model=None,
    
                 condition_classifier_model=None, count_vectorizer=None ):

        self.job_classifier_model = job_classifier_model
        self.general_question_classifier_model = general_question_classifier_model
        self.busy_classifier_model = busy_classifier_model
        self.additional_feature_classifier_model = additional_feature_classifier_model
        self.condition_classifier_model = condition_classifier_model
        self.count_vectorizer = count_vectorizer
        
        
    def predict(self, query):
        query_vectorizer = self.count_vectorizer.transform(pd.Series(lemmatize_sentence(query)))
        general_predict = self.general_question_classifier_model.predict(query_vectorizer)[0]
        if general_predict == 1:
            predictions = {
                "Работа": "None",
                "Занятость": "None",
                "Дополнительный признак": "None",
                "Условия": "None"
            }
        else:
            predictions = {
                "Работа": ", ".join(self.job_classifier_model.predict(query_vectorizer)[0].tolist()),
                "Занятость": pred_labels_for_busy(self.busy_classifier_model.predict(query_vectorizer)[0]),
                "Дополнительный признак": pred_labels_for_additional_feature(self.additional_feature_classifier_model.predict(query_vectorizer)[0]),
                "Условия": ", ".join(self.condition_classifier_model.predict(query_vectorizer)[0].tolist())
            }
        
        return predictions
        
        
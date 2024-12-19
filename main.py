from model import ClassificationModel
import joblib

def main():
    # Загружаем модели из файлов
    catboost_model = joblib.load('models/job_classifier_model.pkl')
    ovr_busy = joblib.load('models/busy_classifier_model.pkl')
    ovr_additional_feature = joblib.load('models/additional_feature_classifier_model.pkl')
    catboost_model_condition = joblib.load('models/condition_classifier_model.pkl')
    catboost_model_general_question = joblib.load('models/general_question_classifier_model.pkl')
    count_vectorizer = joblib.load('models/count_vectorizer.pkl')

    # Создаем объект модели
    model = ClassificationModel(
        job_classifier_model=catboost_model,
        general_question_classifier_model=catboost_model_general_question,
        busy_classifier_model=ovr_busy,
        additional_feature_classifier_model=ovr_additional_feature,
        condition_classifier_model=catboost_model_condition,
        count_vectorizer=count_vectorizer
    )
    
    query = input()
    predictions = model.predict(query)

    # Выводим результаты
    print(f"Запрос: {query}")
    for category, prediction in predictions.items():
        print(f"{category}: {prediction}")

if __name__ == '__main__':
    main()
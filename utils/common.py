import pandas as pd
from pymorphy3 import MorphAnalyzer


def clear_dataset(df: pd.DataFrame):
    df_mc = df.copy().reset_index(drop=True)
    # метки для мультиклассификации
    for col in ["занятость", "по дополнительному признаку"]:
        col_index = df_mc[~df_mc[col].isna()].index
        for ind in col_index:
            descr = df_mc[col].iloc[ind]
            item_list = descr.strip(",. ").lower().split(',')
            clear_list = [item.strip(',. ') for item in item_list if item != ""]
            df_mc.loc[ind, col] = clear_list
    
    # метки для многоклассовой классификации (только к одному принадлежит)
    one_class_cols = ["по должности-лемме", "по условиям"]
    df_mc[one_class_cols] = df_mc[one_class_cols].apply(
        lambda x: x.str.lower()
    )
    df_mc['общие фразы'] = df_mc['общие фразы'].fillna(0).map(
        {"общая фраза": 1, 0: 0}
    )

    return df_mc

def prepare_dataset(data_: pd.DataFrame):
    data = data_.copy()

    for col in ["по должности-лемме", "по условиям"]:
        data[col] = data[col].fillna("None")

    for col in ["занятость", "по дополнительному признаку"]:
        data[col] = data[col].apply(lambda d: d if isinstance(d, list) else [])

    work_type = data.pop("по должности-лемме")
    cond_type = data.pop("по условиям")
    add_type = data.pop("по дополнительному признаку")
    occ_type = data.pop("занятость")
    common_type = data.pop("общие фразы")

    target_cats = {
        "занятость": occ_type,
        "по должности-лемме": work_type,
        "по дополнительному признаку": add_type,
        "общие фразы": common_type,
        "по условиям": cond_type,
    }

    return data['query'], target_cats


def pred_labels_for_busy(binary_vector):
    labels = ['вахта', 'вечерняя', 'временная', 'дневная', 'на дому', 
            'на неполный день', 'ночная', 'по выходным', 'подработка', 
          'посменная', 'посуточная', 'удаленная']
    selected_labels = [labels[i] for i in range(len(binary_vector)) if binary_vector[i] == 1]
    return ", ".join(selected_labels)


def pred_labels_for_additional_feature(binary_vector):
    labels = ['без опыта', 'для женщин', 'для инвалидов', 'для мужчин',
       'для пенсионеров', 'для студентов', 'для школьников']
    selected_labels = [labels[i] for i in range(len(binary_vector)) if binary_vector[i] == 1]
    return ", ".join(selected_labels)


def lemmatize_sentence(sentence):
    morph = MorphAnalyzer()
    words = sentence.split()  
    lemmas = [morph.parse(word)[0].normal_form for word in words]  
    return ' '.join(lemmas)

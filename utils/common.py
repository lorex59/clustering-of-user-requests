import pandas as pd

def clear_dataset(df: pd.DataFrame):
    df_mc = df.copy().reset_index(drop=True)
    # метки для мультиклассификации
    for col in ["занятость", "по дополнительному признаку"]:
        col_index = df_mc[~df_mc[col].isna()].index
        display(df_mc.iloc[col_index][col].isna().sum())
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

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
sns.set()


column_names = {
  1: "ANNUAL INCOME OF HOUSEHOLD (PERSONAL INCOME IF SINGLE)",
  2: "SEX",
  3: "MARITAL STATUS",
  4: "AGE",
  5: "EDUCATION",
  6: "OCCUPATION",
  7: "HOW LONG HAVE YOU LIVED IN THE SAN FRAN./OAKLAND/SAN JOSE AREA?",
  8: "DUAL INCOMES (IF MARRIED)",
  9: "PERSONS IN YOUR HOUSEHOLD",
  10: "PERSONS IN HOUSEHOLD UNDER 18",
  11: "HOUSEHOLDER STATUS",
  12: "TYPE OF HOME",
  13: "ETHNIC CLASSIFICATION",
  14: "WHAT LANGUAGE IS SPOKEN MOST OFTEN IN YOUR HOME?"
}

column_names_rename = {
    '1' : 'income',
    '2' : 'sex',
    '3' : 'marital status',
    '4' : 'age',
    '5' : 'education',
    '6' : 'occupation',
    '7' : 'years in bay area',
    '8' : 'dual incomes',
    '9' : 'num persons hh',
    '10': 'num children hh',
    '11': 'hh status',
    '12': 'hometype',
    '13': 'ethnicity',
    '14': 'language'
}


def prepare_plot(size_x, size_y):
    return plt.subplots(1, 1, figsize=(size_x, size_y), sharex=True)


def prepare_plots(size_x, size_y, grid_x, grid_y):
    return plt.subplots(grid_x, grid_y, figsize=(size_x, size_y), sharex=True)


def violin_plot(data, col1, col2, return_fig=False,
                sns_inner='point', sns_scale='count', sns_sex='both-split'):
    if col1 == col2:
        return None
    
    hue = '2'
    sns_split = True if sns_sex == 'both-split' else False
    sns_hue = hue if sns_split else None
    if sns_sex in ['male', 'female']:
        data = data[data[hue] == { 'male': 1, 'female': 2 }[sns_sex]]
    data = data.dropna(subset=[col1, col2])

    fig, ax = prepare_plot(10, 7)
    # fig, ax = prepare_plot(16, 10)
    ax = sns.violinplot(x=col1, y=col2, data=data, palette='Set2',
                        hue=sns_hue, split=sns_split,
                        inner=sns_inner, scale=sns_scale)

    x_ticks = { int(float(i.get_text())) : i.get_position()[0] for i in ax.get_xticklabels() }
    for i in data[col1].unique():
        if sns_sex == 'both-split':
            hue_count_1 = len( data[(data[col1] == i) & (data[hue] == 1)] )
            hue_count_2 = len( data[(data[col1] == i) & (data[hue] == 2)] )
            counts = f'{hue_count_1} / {hue_count_2}'
        else:
            counts = f'{len( data[data[col1] == i] )}'
        ax.text(x_ticks[i] - 0.25, 0, counts,
                bbox=dict(color='white', boxstyle='round') )
    ax.set(xlabel=f'{col1} | {column_names_rename[col1]}', ylabel=f'{col2} | {column_names_rename[col2]}')    
    
    if return_fig:
        return fig


def violin_plot_widget(data):
    widgets.interact(violin_plot,
                 return_fig=widgets.fixed(False),
                 data=widgets.fixed(data),
                 col1=data.columns,
                 col2=data.columns,
                 sns_inner = ['point', 'quartile', 'box', 'stick', None],
                 sns_scale = ['count', 'area', 'width'],
                 sns_sex=['both-split', 'both-combo', 'male', 'female'])


def get_nan_counts(df):
    nan_counts = df.rename({str(k): f'{k} | {v}' for k, v in column_names_rename.items()}, axis=1)
    nan_counts = nan_counts.isna().sum()
    nan_counts = pd.concat([nan_counts, nan_counts / len(df) * 100], axis=1)
    nan_counts.columns = ['Count', 'Percentage']
    return nan_counts


def get_correlations(df, drop_duplicates=True):
    corr = (df
        .corr()
        .abs()
        .unstack()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'level_0': 'F1', 'level_1': 'F2', 0: 'Correlation'})
        .query('Correlation != 1')
    )
    if drop_duplicates:
        corr['set'] = corr.apply(lambda x: set([x['F1'], x['F2']]), axis=1).astype(str)
        corr = corr.drop_duplicates('set').drop(columns='set')
    return corr


def reorder_around_feature(df, label='1', skip_features=['2']):
    reordered = {}
    for col in df.columns.drop([label] + skip_features):
        result = (df
            .groupby(col)
            .median()[label]
            .sort_values()
            .index
            .tolist()
        )
        reordered[col] = dict(zip(np.unique(result), result))
    reordered_back = { k: { vv: kk for kk, vv in v.items() } for k, v in reordered.items() }
    return df.replace(reordered), reordered, reordered_back


def catboost_fill(df, features, label, depth_select='auto'):
    def catboost_eval(df, features, label, depth):
        df_clean = df.dropna(subset=features)
        df_to_restore = df[~df.index.isin(df_clean.index)]
        
        df_for_model = df_clean[features + [label]]
        df_train_test = df_for_model[df_for_model[label].notnull()].astype(int)
        df_pred = df_for_model[df_for_model[label].isnull()]

        mask_train_test = np.random.rand(len(df_train_test)) < 0.8
        df_train, df_test = df_train_test[mask_train_test], df_train_test[~mask_train_test]

        x_train, y_train = df_train.drop(columns=label), df_train[label]
        x_test, y_test = df_test.drop(columns=label), df_test[label]
        x_pred = df_pred.drop(columns=label)
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=depth,
            loss_function='MultiClass',
            early_stopping_rounds=200,
            verbose=False
        ).fit(x_train, y_train, eval_set=(x_test, y_test))
        acc = (model.predict(x_test).flatten() == y_test.values).sum() / len(x_test)

        y_pred = model.predict(x_pred).flatten()
        df_clean[label][df_clean[label].isnull()] = y_pred
        return pd.concat([df_clean, df_to_restore]), acc
    
    if features is None:
        corr = get_correlations(df, drop_duplicates=False)
        features = corr[corr['F1'] == label].head(5)['F2'].tolist()

    if depth_select == 'auto':
        acc = 0
        result = None
        for depth in range(1, len(features) + 1):
            result_next, acc_next = catboost_eval(df, features, label, depth)
            if acc_next > acc:
                acc = acc_next
                result = result_next
        msg = f'Accuracy: {acc} with depth {depth} for features {features}'
        print(msg)
        return result, msg
    else:
        depth = len(features)
        if depth_select >= 0:
            depth = depth // 2 + max(depth, depth_select)
            depth = depth if depth != 0 else 1
        return catboost_eval(df, features, label, depth + 1)

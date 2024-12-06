import pandas as pd
import matplotlib.pyplot as plt

def recognize_id_split(ids_train, ids_test):
    ids_train = set(ids_train)
    ids_test = set(ids_test)
    
    ids_test_only = ids_test - ids_train
    ids_joint = ids_train.intersection(ids_test)
    id_split = 'closed-set'
    if len(ids_joint) == 0:
        id_split = 'disjoint-set'
    elif len(ids_test_only) > 0:
        id_split = 'open-set'
    return id_split

def recognize_time_split(df_train, df_test, col_label='identity'):
    ids_train = set(df_train[col_label])
    ids_test = set(df_test[col_label])
    ids_joint = ids_train.intersection(ids_test)

    time_split = 'time-unaware'
    if 'date' in df_train.columns:
        if max(df_train['date']) <= min(df_test['date']):
            time_split = 'time-cutoff'                
        else:
            change_split = True
            for identity in ids_joint:
                df_train_red = df_train[df_train[col_label] == identity]
                df_test_red = df_test[df_test[col_label] == identity]
                if len(df_train_red) > 0 and len(df_test_red) > 0 and max(df_train_red['date']) > min(df_test_red['date']):
                    change_split = False
                    break
            if change_split and len(ids_joint) > 0:
                time_split = 'time-proportion'
    return time_split

def extract_data_split(df, idx_train, idx_test, col_label='identity'):
    df_train = df.loc[idx_train]
    df_test = df.loc[idx_test]
    
    ids = set(df[col_label])
    ids_train = set(df_train[col_label])
    ids_test = set(df_test[col_label])
    ids_train_only = ids_train - ids_test
    ids_test_only = ids_test - ids_train
    
    id_split = recognize_id_split(ids_train, ids_test)
    time_split = recognize_time_split(df_train, df_test)

    return {
        'id_split': id_split,
        'time_split': time_split,
        'n': len(df),
        'n_train': len(df_train),
        'n_test': len(df_test),
        'n_train_only': sum([sum(df_train[col_label] == ids) for ids in ids_train_only]),
        'n_test_only': sum([sum(df_test[col_label] == ids) for ids in ids_test_only]),
        'n_unassigned': len(df)-len(df_train)-len(df_test),
        'n_ids': len(ids),
        'n_ids_train': len(ids_train),
        'n_ids_test': len(ids_test),
        'n_ids_joint': len(ids_train.intersection(ids_test)),
        'n_ids_train_only': len(ids_train_only),
        'n_ids_test_only': len(ids_test_only),
        'n_ids_unassigned': len(ids)-len(ids_train)-len(ids_test)+len(ids_train.intersection(ids_test)),
    }

def analyze_split(df, idx_train, idx_test, col_label='identity'):
    data = extract_data_split(df, idx_train, idx_test, col_label=col_label)
    print('Split: %s %s' % (data['time_split'], data['id_split']))
    print('Samples: train/test/unassigned/total = %d/%d/%d/%d' % (data['n_train'], data['n_test'], data['n_unassigned'], data['n']))
    print('Classes: train/test/unassigned/total = %d/%d/%d/%d' % (data['n_ids_train'], data['n_ids_test'], data['n_ids_unassigned'], data['n_ids']))
    print('Samples: train only/test only        = %d/%d' % (data['n_train_only'], data['n_test_only']))
    print('Classes: train only/test only/joint  = %d/%d/%d' % (data['n_ids_train_only'], data['n_ids_test_only'],  data['n_ids_joint']))
    print('')    
    print('Fraction of train set     = %1.2f%%' % (100*data['n_train']/data['n']))
    print('Fraction of test set only = %1.2f%%' % (100*data['n_test_only']/data['n']))

def visualize_split(df_train, df_test, selection=None, ylabels=True, col_label='identity'):
    if selection == None:
        selection = pd.concat((df_train, df_test))[col_label].sort_values().unique()
    elif isinstance(selection, int):
        selection = pd.concat((df_train, df_test))[col_label].sort_values().unique()[:selection]
    for i, identity in enumerate(selection):
        date_train = df_train[df_train[col_label] == identity]['date']
        plt.scatter(date_train, len(date_train)*[i], color='blue')
        date_test = df_test[df_test[col_label] == identity]['date']
        plt.scatter(date_test, len(date_test)*[i], color='black', marker='*')
    if ylabels:
        plt.yticks(range(len(selection)), selection)
        plt.ylabel('Turtle name')
    plt.scatter([], [], color='blue', label='Training set')
    plt.scatter([], [], color='black', marker='*', label='Testing set')
    plt.legend()
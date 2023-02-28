def analyze_split(df, idx_train, idx_test):
    df_train = df.loc[idx_train]
    df_test = df.loc[idx_test]
    
    ids = set(df['identity'])
    ids_train = set(df_train['identity'])
    ids_test = set(df_test['identity'])
    ids_train_only = ids_train - ids_test
    ids_test_only = ids_test - ids_train
    ids_joint = ids_train.intersection(ids_test)
    
    n = len(idx_train)+len(idx_test)
    n_train = len(idx_train)
    n_test_only = sum([sum(df_test['identity'] == ids) for ids in ids_test_only])    
    
    ratio_train = n_train / n    
    ratio_test_only = n_test_only / n
    
    time_split = 'Time-unaware'
    if 'date' in df.columns:
        if max(df_train['date']) <= min(df_test['date']):
            time_split = 'Time-cutoff'                
        else:
            change_split = True
            for identity in ids_joint:
                df_train_red = df_train[df_train['identity'] == identity]
                df_test_red = df_test[df_test['identity'] == identity]
                if len(df_train_red) > 0 and len(df_test_red) > 0 and max(df_train_red['date']) > min(df_test_red['date']):
                    change_split = False
                    break
            if change_split and len(ids_joint) > 0:
                time_split = 'Time-proportion'
                
    id_split = 'Closed-set'
    if len(ids_joint) == 0:
        id_split = 'Disjoint-set'
    elif len(ids_test_only) > 0:
        id_split = 'Open-set'
            
    print('%s %s split' % (time_split, id_split.lower()))    
    print('Samples: train/test/unassigned/total = %d/%d/%d/%d' % (len(df_train), len(df_test), len(df)-len(df_train)-len(df_test), len(df)))
    print('Classes: train/test/unassigned/total = %d/%d/%d/%d' % (len(ids_train), len(ids_test), len(ids)-len(ids_train)-len(ids_test)+len(ids_train.intersection(ids_test)), len(ids)))
    print('Classes: train only/test only/joint  = %d/%d/%d' % (len(ids_train_only), len(ids_test_only),  len(ids_joint)))
    print('')    
    print('Fraction of train set     = %1.2f%%' % (100*ratio_train))
    print('Fraction of test set only = %1.2f%%' % (100*ratio_test_only))
import os
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    # KFold
    input_path = '../data/'
    df = pd.read_csv(os.path.join(input_path, 'labels.csv'))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.xRot.values
    kf = model_selection.KFold(n_splits=5)
    for fold_num, (train_index, test_index) in enumerate(kf.split(X=df, y=y)):
        df.iloc[test_index, df.columns.get_loc('kfold')] = fold_num
        df['is_valid'] = 'False'
        df.iloc[test_index, df.columns.get_loc('is_valid')] = 'True'
        df.to_csv(os.path.join(
            input_path, f'train_fold_{fold_num}.csv'), index=False)

    df.to_csv(os.path.join(input_path, 'labels_folds.csv'), index=False)

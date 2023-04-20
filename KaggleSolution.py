import cluster as cluster
import metrics as metrics
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn import *
import glob
from sklearn.base import clone
import pathlib
from seglearn.feature_functions import base_features, emg_features
from sklearn.model_selection import GroupKFold
from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
from tsflex.features.integrations import seglearn_feature_dict_wrapper
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform, randint
from sklearn.metrics import average_precision_score, make_scorer


p = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/'

train = glob.glob(p+'train/**/**')
test = glob.glob(p+'test/**/**')
subjects = pd.read_csv(p+'subjects.csv')
tasks = pd.read_csv(p+'tasks.csv')
sub = pd.read_csv(p+'sample_submission.csv')

tdcsfog_metadata=pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv')
defog_metadata=pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/defog_metadata.csv')
tdcsfog_metadata['Module']='tdcsfog'
defog_metadata['Module']='defog'
metadata=pd.concat([tdcsfog_metadata,defog_metadata])
print(metadata)

# https://www.kaggle.com/code/jazivxt/familiar-solvs
tasks['Duration'] = tasks['End'] - tasks['Begin']
tasks = pd.pivot_table(tasks, values=['Duration'], index=['Id'], columns=['Task'], aggfunc='sum', fill_value=0)
tasks.columns = [c[-1] for c in tasks.columns]
tasks = tasks.reset_index()
tasks['t_kmeans'] = cluster.KMeans(n_clusters=10, random_state=3).fit_predict(tasks[tasks.columns[1:]])

subjects = subjects.fillna(0).groupby('Subject').median()
subjects = subjects.reset_index()
# subjects.rename(columns={'Subject':'Id'}, inplace=True)
subjects['s_kmeans'] = cluster.KMeans(n_clusters=10, random_state=3).fit_predict(subjects[subjects.columns[1:]])
subjects=subjects.rename(columns={'Visit':'s_Visit','Age':'s_Age','YearsSinceDx':'s_YearsSinceDx','UPDRSIII_On':'s_UPDRSIII_On','UPDRSIII_Off':'s_UPDRSIII_Off','NFOGQ':'s_NFOGQ'})

print(tasks)
print(subjects)

complex_featlist=['Visit','Test','Medication','s_Visit','s_Age','s_YearsSinceDx','s_UPDRSIII_On','s_UPDRSIII_Off','s_NFOGQ','s_kmeans']
metadata_complex=metadata.merge(subjects,how='left',on='Subject').copy()
metadata_complex['Medication']=metadata_complex['Medication'].factorize()[0]

print(metadata_complex)



basic_feats = MultipleFeatureDescriptors(
    functions=seglearn_feature_dict_wrapper(base_features()),
    series_names=['AccV', 'AccML', 'AccAP'],
    windows=[5_000],
    strides=[5_000],
)

emg_feats = emg_features()
del emg_feats['simple square integral'] # is same as abs_energy (which is in base_features)

emg_feats = MultipleFeatureDescriptors(
    functions=seglearn_feature_dict_wrapper(emg_feats),
    series_names=['AccV', 'AccML', 'AccAP'],
    windows=[5_000],
    strides=[5_000],
)

fc = FeatureCollection([basic_feats, emg_feats])


def reader(f):
    try:
        df = pd.read_csv(f, index_col="Time",
                         usecols=['Time', 'AccV', 'AccML', 'AccAP', 'StartHesitation', 'Turn', 'Walking'])

        df['Id'] = f.split('/')[-1].split('.')[0]
        df['Module'] = pathlib.Path(f).parts[-2]
        df = pd.merge(df, tasks[['Id', 't_kmeans']], how='left', on='Id').fillna(-1)
        #         df = pd.merge(df, subjects[['Id','s_kmeans']], how='left', on='Id').fillna(-1)
        df = pd.merge(df, metadata_complex[['Id', 'Subject'] + ['Visit', 'Test', 'Medication', 's_kmeans']], how='left',
                      on='Id').fillna(-1)
        df_feats = fc.calculate(df, return_df=True, include_final_window=True, approve_sparsity=True,
                                window_idx="begin").astype(np.float32)
        df = df.merge(df_feats, how="left", left_index=True, right_index=True)
        df.fillna(method="ffill", inplace=True)
        return df
    except:
        pass


train = pd.concat([reader(f) for f in tqdm(train)]).fillna(0);
print(train.shape)
cols = [c for c in train.columns if
        c not in ['Id', 'Subject', 'Module', 'Time', 'StartHesitation', 'Turn', 'Walking', 'Valid', 'Task', 'Event']]
pcols = ['StartHesitation', 'Turn', 'Walking']
scols = ['Id', 'StartHesitation', 'Turn', 'Walking']



best_params_ = {'estimator__colsample_bytree': 0.5282057895135501,
                'estimator__learning_rate': 0.15,
                'estimator__max_depth': 8,
                'estimator__min_child_weight': 3.1233911067827616,
                'estimator__n_estimators': 291,
                'estimator__subsample': 0.999}  # 0.9961057796456088
best_params_ = {kk: v for k, v in best_params_.items() for kk in k.split('__')};
del best_params_['estimator']



def custom_average_precision(y_true, y_pred):
    score = average_precision_score(y_true, y_pred)
    return 'average_precision', score, True


class LGBMMultiOutputRegressor(MultiOutputRegressor):
    def fit(self, X, y, eval_set=None, **fit_params):
        self.estimators_ = [clone(self.estimator) for _ in range(y.shape[1])]

        for i, estimator in enumerate(self.estimators_):
            if eval_set:
                fit_params['eval_set'] = [(eval_set[0], eval_set[1][:, i])]
            estimator.fit(X, y[:, i], **fit_params)

        return self




N_FOLDS = 5
kfold = GroupKFold(N_FOLDS)
group_var = train.Subject
groups = kfold.split(train, groups=group_var)
regs = []
cvs = []
for fold, (tr_idx, te_idx) in enumerate(tqdm(groups, total=N_FOLDS, desc="Folds")):
    tr_idx = pd.Series(tr_idx).sample(n=2000000, random_state=42).values  # 2000000

    # Create a base XGBoost regressor with the common parameters
    base_regressor = lgb.LGBMRegressor(**best_params_)

    # Wrap the base regressor with the MultiOutputRegressor
    multioutput_regressor = LGBMMultiOutputRegressor(base_regressor)

    x_tr, y_tr = train.loc[tr_idx, cols].to_numpy(), train.loc[tr_idx, pcols].to_numpy()
    x_te, y_te = train.loc[te_idx, cols].to_numpy(), train.loc[te_idx, pcols].to_numpy()

    multioutput_regressor.fit(
        x_tr, y_tr,
        eval_set=(x_te, y_te),
        eval_metric=custom_average_precision,
        early_stopping_rounds=25
    )
    regs.append(multioutput_regressor)
    cv = metrics.average_precision_score(y_te, multioutput_regressor.predict(x_te).clip(0.0, 1.0))
    cvs.append(cv)
print(cvs)

sub['t'] = 0
submission = []
for f in test:
    df = pd.read_csv(f)
    df.set_index('Time', drop=True, inplace=True)

    df['Id'] = f.split('/')[-1].split('.')[0]
    #     df = df.fillna(0).reset_index(drop=True)
    df = pd.merge(df, tasks[['Id', 't_kmeans']], how='left', on='Id').fillna(-1)
    #     df = pd.merge(df, subjects[['Id','s_kmeans']], how='left', on='Id').fillna(-1)
    df = pd.merge(df, metadata_complex[['Id', 'Subject'] + ['Visit', 'Test', 'Medication', 's_kmeans']], how='left',
                  on='Id').fillna(-1)
    df_feats = fc.calculate(df, return_df=True, include_final_window=True, approve_sparsity=True, window_idx="begin")
    df = df.merge(df_feats, how="left", left_index=True, right_index=True)
    df.fillna(method="ffill", inplace=True)
    #     res = pd.DataFrame(np.round(reg.predict(df[cols]).clip(0.0,1.0),3), columns=pcols)

    res_vals = []
    for i_fold in range(N_FOLDS):
        res_val = np.round(regs[i_fold].predict(df[cols]).clip(0.0, 1.0), 3)
        res_vals.append(np.expand_dims(res_val, axis=2))
    res_vals = np.mean(np.concatenate(res_vals, axis=2), axis=2)
    res = pd.DataFrame(res_vals, columns=pcols)

    df = pd.concat([df, res], axis=1)
    df['Id'] = df['Id'].astype(str) + '_' + df.index.astype(str)
    submission.append(df[scols])
submission = pd.concat(submission)
submission = pd.merge(sub[['Id']], submission, how='left', on='Id').fillna(0.0)
submission[scols].to_csv('submission.csv', index=False)

print(submission)
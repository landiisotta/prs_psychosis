import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def gpc_scalinig():
    feat90 = pd.read_csv('../out/rescale_feature_opcrit90.txt')

    opcrit90_files = ["gpc_clinical_opcrit90", "gpc_all_opcrit90", "gpc_genetic_opcrit90"]

    scaler = MinMaxScaler()
    for f90 in opcrit90_files:
        train = pd.read_csv(f'../out/{f90}_train.txt', sep='\t')
        test = pd.read_csv(f'../out/{f90}_test.txt', sep='\t')
        cols = sorted(list(set(feat90.feat).intersection(set(train.columns))))

        opcrit90_train = pd.DataFrame(columns=train.columns)
        opcrit90_test = pd.DataFrame(columns=test.columns)
        for anc in ['eur', 'afr', 'amr']:
            train_anc = train.loc[train.ancestry == anc].copy().reset_index(drop=True)
            train_tmp = pd.DataFrame(scaler.fit_transform(train_anc[cols]),
                                     columns=train_anc[cols].columns)
            train_anc.update(train_tmp, overwrite=True)
            opcrit90_train = opcrit90_train.append(train_anc, ignore_index=True)

            test_anc = test.loc[test.ancestry == anc].copy().reset_index(drop=True)
            test_tmp = pd.DataFrame(scaler.transform(test_anc[cols]),
                                    columns=test_anc[cols].columns)
            test_anc.update(test_tmp, overwrite=True)
            opcrit90_test = opcrit90_test.append(test_anc, ignore_index=True)

        opcrit90_train = opcrit90_train.astype({'OPCRIT.90': 'int32'})
        opcrit90_test = opcrit90_test.astype({'OPCRIT.90': 'int32'})
        opcrit90_train.to_csv(f'../out/{f90}_scaled_train.txt', sep='\t', index=False)
        opcrit90_test.to_csv(f'../out/{f90}_scaled_test.txt', sep='\t', index=False)


if __name__ == '__main__':
    gpc_scalinig()

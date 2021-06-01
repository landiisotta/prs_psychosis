import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools


def scale_biome():
    feat = pd.read_csv('../out/rescale_feature_biome.txt')

    sh_files = ["biome_all_", "biome_genetic_"]
    names = ["agressive", "psych_admit"]

    scaler = MinMaxScaler()
    for f in itertools.product(sh_files, names):
        train = pd.read_csv(f'../out/{f[0]}{f[1]}_train.txt', sep='\t')
        test = pd.read_csv(f'../out/{f[0]}{f[1]}_test.txt', sep='\t')
        cols = sorted(list(set(feat.feat).intersection(set(train.columns))))

        train_df = pd.DataFrame(columns=train.columns)
        test_df = pd.DataFrame(columns=test.columns)
        for anc in ["EUR", "AFR", "AMR"]:
            train_anc = train.loc[train['gill.ContinentalGrouping'] == anc].copy().reset_index(drop=True)
            train_tmp = pd.DataFrame(scaler.fit_transform(
                train_anc[cols]), columns=train_anc[cols].columns)
            train_anc.update(train_tmp, overwrite=True)
            train_df = train_df.append(train_anc, ignore_index=True)

            test_anc = test.loc[test['gill.ContinentalGrouping'] == anc].copy().reset_index(drop=True)
            test_tmp = pd.DataFrame(scaler.transform(test_anc[cols]), columns=test_anc[cols].columns)
            test_anc.update(test_tmp, overwrite=True)
            test_df = test_df.append(test_anc, ignore_index=True)

        train_df.to_csv(f'../out/{f[0]}{f[1]}_scaled_train.txt', sep='\t', index=False)
        test_df.to_csv(f'../out/{f[0]}{f[1]}_scaled_test.txt', sep='\t', index=False)


if __name__ == '__main__':
    scale_biome()

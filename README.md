## Learning curves.

This repo contains code to generate learning curves. The primary application is the supervised problem of predicting response of cancer cell lines to anti-cancer drugs. 

There are three main scripts you need to run:
(1) Genrate topN dataset (build_topN.py)
(2) Genrate the data splits (gen_data_splits.py)
(3) Use data splits from (2) to generate learning curves (main_lrn_crv.py)

Script (1) requires to have a folder called "data" that contains a set of required files.
You can just copy the folder /vol/ml/apartin/projects/candle/data to your parent dir.

### Step-by-step runs
(1) First, run script (1) as follows:
```py
python build_topN.py --top_n 6 --format parquet --labels
```
This will create dir called top6_data. The folder will contain a single parquet file. In addition, some plots are generated.

(2) Then, run script (2) to generate the data splits:
```py
python gen_data_splits.py --dirpath top6_data
```
This will create dir called top6_data_splits that contains splits for various k-folds. In addition data files are generated xdata.parquet and meta.parquet.

(3) Finally, run script (3) to generate the learning curves:
```py
python main_lrn_crv.py --dirpath top6_data --clr_mode trng1
```

## Learning curves.

This repo contains code to generates learning curves. The primary application is the supervised problem of predicting the response of cancer cell lines to anti-cancer drugs. 

There three main scripts you need to run:
(1) Script that genrates topN dataset and the data splits (build_topN.py)
(2) Script that genrates the data splits (gen_data_splits.py)
(3) Script that uses the data splits to generate the learning curves (main_lrn_crv.py)

Script (1) requires to have a folder called "data" that contains some required files.
You can just copy the folder /vol/ml/apartin/projects/candle/data to your parent dir.

### Step-by-step runs
(1) First, run script (1) as follows:
```py
python build_topN.py --top_n 6 --format parquet --labels
```
This will create dir called top6_data

(2) Then, run script (2) to generate the data splits:
```py
python gen_data_splits.py --dirpath top6_data
```
This will create dir called top6_data_splits

(3) Finally, run script (3) to generate the learning curves:
```py
python main_lrn_crv.py --dirpath top21_data --clr_mode trng1
```
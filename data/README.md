# Data

Data is **not** included in this repository: the RSNA Pneumonia Detection Challenge rules do not allow redistribution.

## How to get it

1. Create a Kaggle account and accept the rules of the competition:
   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
2. Either attach the competition data to a Kaggle Notebook (recommended), or download it locally with the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/
unzip data/rsna-pneumonia-detection-challenge.zip -d data/rsna
```

`src.data.find_data_root()` locates the folder automatically (it searches for `stage_2_train_labels.csv`).

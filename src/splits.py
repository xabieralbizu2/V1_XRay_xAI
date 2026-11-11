# src/splits.py
from sklearn.model_selection import train_test_split

def make_splits(df, val_size=0.15, test_size=0.15, seed=42):
    """Patient-level split stratified by target and view."""
    df = df[(df.age > 0) & (df.age <= 120)].copy()
    strata = df.target.astype(str) + "_" + df.view
    train_val, test = train_test_split(df, test_size=test_size, stratify=strata, random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val,
                                  stratify=strata.loc[train_val.index], random_state=seed)
    return train.assign(split="train"), val.assign(split="val"), test.assign(split="test")
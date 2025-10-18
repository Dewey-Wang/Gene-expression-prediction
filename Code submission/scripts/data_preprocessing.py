import pandas as pd
def load_and_concat(train_path, val_path):
    df_train = pd.read_csv(train_path, sep="\t")
    df_val = pd.read_csv(val_path, sep="\t")
    df = pd.concat([df_train, df_val], ignore_index=True)
    return df


def pick_reference(row):
    tss_start_1, tss_start_2 = row.get("TSS_start_X1"), row.get("TSS_start_X2")
    tss_end_1, tss_end_2 = row.get("TSS_end_X1"), row.get("TSS_end_X2")

    tss_start_1 = pd.to_numeric(tss_start_1, errors="coerce")
    tss_start_2 = pd.to_numeric(tss_start_2, errors="coerce")
    tss_end_1 = pd.to_numeric(tss_end_1, errors="coerce")
    tss_end_2 = pd.to_numeric(tss_end_2, errors="coerce")

    if pd.notna(tss_start_1) and pd.notna(tss_start_2):
        tss_start = min(tss_start_1, tss_start_2)
    else:
        tss_start = tss_start_1 if pd.notna(tss_start_1) else tss_start_2

    if pd.notna(tss_end_1) and pd.notna(tss_end_2):
        tss_end = max(tss_end_1, tss_end_2)
    else:
        tss_end = tss_end_1 if pd.notna(tss_end_1) else tss_end_2

    return pd.Series({"TSS_start": int(tss_start), "TSS_end": int(tss_end)})

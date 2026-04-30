def split(df, train=0.7, val=0.8, time_col="Date Time"):

    # 1. Define the split ratios
    n = len(df)
    train_end = int(n * train)
    val_end = int(n * val)  # 70% to 80% is the 10% validation slice

    # 2. Split the dataframe
    # We keep the raw data for now; we'll handle normalization next
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Total records: {n}")
    print(f"Train set: {len(train_df)} rows")
    print(f"Val set:   {len(val_df)} rows")
    print(f"Test set:  {len(test_df)} rows")

    # 3. Separate features from the timestamp
    # The 'Date Time' column isn't a feature the model processes directly
    # We'll need it for plotting later, but for the Transformer, we focus on
    # the values.
    cols_to_drop = [time_col]
    features = [c for c in df.columns if c not in cols_to_drop]

    print(f"\nNumber of features to be used: {len(features)}")

    train_data = train_df[features]
    val_data = val_df[features]
    test_data = test_df[features]

    return train_data, val_data, test_data, features

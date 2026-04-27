def norm(train_df, val_df, test_df, features):
    # 1. Identify the feature columns (excluding the timestamp)
    feature_cols = features

    # 2. Calculate mean and std from the TRAINING set only
    train_mean = train_df[feature_cols].mean()
    train_std = train_df[feature_cols].std()

    # 3. Apply the transformation: (x - mean) / std
    # We convert to values (numpy) here to prepare for tensor conversion later
    train_data = (train_df[feature_cols] - train_mean) / train_std
    val_data = (val_df[feature_cols] - train_mean) / train_std
    test_data = (test_df[feature_cols] - train_mean) / train_std

    return train_data, val_data, test_data

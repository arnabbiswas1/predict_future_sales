def get_data_splits_by_fraction(dataframe, valid_fraction=0.1):
    """
    Creating holdout set from the train data based on fraction
    """
    print(f'Splitting the data into train and holdout with validation fraction {valid_fraction}...')
    valid_size = int(len(dataframe) * valid_fraction)
    train = dataframe[:valid_size]
    validation = dataframe[valid_size:]
    print(f'Shape of the training data {train.shape} ')
    print(f'Shape of the validation data {validation.shape}')
    return train, validation


def get_data_splits_by_month(dataframe, train_months, validation_months):
    """
    Creating holdout set from the train data based on months
    """
    print(f'Splitting the data into train and holdout based on months...')
    print(f'Training months {train_months}')
    print(f'Validation months {validation_months}')
    training = dataframe[dataframe.month.isin(train_months)]
    validation = dataframe[dataframe.month.isin(validation_months)]
    print(f'Shape of the training data {training.shape} ')
    print(f'Shape of the validation data {validation.shape}')
    return training, validation
import pandas as pd

def concat_features(source_df, target_df, f1, f2):
    print(f'Concating features {f1} and {f2}')
    target_df[f'{f1}_{f2}'] =  source_df[f1].astype(str) + '_' + source_df[f2].astype(str)
    return target_df


def create_interaction_features(source_df, target_df):
    """
    For ASHARE
    """
    print('Creating interaction features...')
    target_df = concat_features(source_df, target_df, 'site_id', 'building_id')
    target_df['site_building_meter_id'] = source_df.site_id.astype(str) + '_' + source_df.building_id.astype(str) + '_' + source_df.meter.astype(str)
    target_df['site_building_meter_id_usage'] = source_df.site_id.astype(str) + '_' + source_df.building_id.astype(str) + '_' + source_df.meter.astype(str) + '_' + source_df.primary_use

    target_df = concat_features(source_df, target_df, 'site_id', 'meter')
    target_df = concat_features(source_df, target_df, 'building_id', 'meter')
    
    target_df = concat_features(source_df, target_df, 'site_id', 'primary_use')
    target_df = concat_features(source_df, target_df, 'building_id', 'primary_use')
    target_df = concat_features(source_df, target_df, 'meter', 'primary_use')
    
    return target_df


def create_age(source_df, target_df, f):
    """
    For ASHARE
    """
    print('Creating age feature')
    target_df['building_age'] = 2019 - source_df[f]
    return target_df
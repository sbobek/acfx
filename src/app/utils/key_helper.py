def calc_pbounds(data, categorical_indicator):
    return {f: (data[f].min(), data[f].max()) for f in data.columns if f not in categorical_indicator}

def get_pbounds_key(feature_name):
    return f"{feature_name}_pbounds"
def get_masked_feature_key(feature_name):
    return f"{feature_name}_is_masked"
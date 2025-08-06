from src.acfx.benchmark.data.consts import TEST_SIZE, RANDOM_STATE
import json

class DataRecipe:
    def __init__(self, target=None, features=None, features_num=None, features_cat=None, features_types=None, name=None,
                 test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.target = target
        self.features = features
        self.features_num = features_num
        self.features_cat = features_cat
        self.features_types = features_types
        self.name = name
        self.test_size = test_size
        self.random_state = random_state

    @classmethod
    def from_json_file(cls, filepath):
        data = json.load(open(filepath, 'r'))
        return cls(
            data.get('target'),
            data.get('features'),
            data.get('features_num'),
            data.get('features_cat'),
            data.get('features_types'),
            data.get('name', filepath[filepath.rfind('/') + 1:filepath.rfind('.')]),
            data.get('test_size', TEST_SIZE),
            data.get('random_state', RANDOM_STATE)
        )
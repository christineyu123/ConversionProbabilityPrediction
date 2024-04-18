from a_preprocessing.explore_preprocessor import ExplorePreprocessor
from a_preprocessing.naive_preprocessor import NaivePreprocessor


def get_preprocessors(preprocessors_names, train_data_path, assessment_data_path):
    all_preprocessors = []
    for name in preprocessors_names:
        if name == "naive":
            cur_preprocessor = NaivePreprocessor(train_data_path=train_data_path,
                                                 assessment_data_path=assessment_data_path)
        elif name == 'explore':
            cur_preprocessor = ExplorePreprocessor(train_data_path=train_data_path,
                                                   assessment_data_path=assessment_data_path)
        else:
            raise ValueError("Unknown preprocessor")
        all_preprocessors.append(cur_preprocessor)

    return all_preprocessors

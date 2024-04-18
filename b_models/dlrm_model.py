import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from a_preprocessing.abstract_preprocessor import PreprocessorSettings
from b_models.abstract_model import Model
from metrics.metric import get_all_metrics, return_counts_y_pred


def MLP(arch, activation='relu', out_activation=None, add_extra_reg=False):
    mlp = tf.keras.Sequential()

    for units in arch[:-1]:
        if add_extra_reg:
            mlp.add(
                tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=tf.keras.regularizers.L2(0.01)))
            mlp.add(tf.keras.layers.BatchNormalization())
            mlp.add(tf.keras.layers.Dropout(0.2))
        else:
            mlp.add(tf.keras.layers.Dense(units, activation=activation))

    mlp.add(tf.keras.layers.Dense(arch[-1], activation=out_activation))

    return mlp


class SecondOrderFeatureInteraction(tf.keras.layers.Layer):
    def __init__(self, self_interaction=False):
        super(SecondOrderFeatureInteraction, self).__init__()
        self.self_interaction = self_interaction

    def call(self, inputs):
        batch_size = tf.shape(inputs[0])[0]
        concat_features = tf.stack(inputs, axis=1)

        dot_products = tf.matmul(concat_features, concat_features, transpose_b=True)

        ones = tf.ones_like(dot_products)
        mask = tf.linalg.band_part(ones, 0, -1)
        out_dim = int(len(inputs) * (len(inputs) + 1) / 2)

        if not self.self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)
            out_dim = int(len(inputs) * (len(inputs) - 1) / 2)

        flat_interactions = tf.reshape(tf.boolean_mask(dot_products, mask), (batch_size, out_dim))
        return flat_interactions


class DLRM(tf.keras.Model):
    def __init__(
            self,
            embedding_sizes,
            embedding_dim,
            arch_bot,
            arch_top,
            self_interaction,
    ):
        super(DLRM, self).__init__()
        self.emb = [tf.keras.layers.Embedding(size, embedding_dim) for size in embedding_sizes]
        self.bot_nn = MLP(arch_bot, out_activation='relu')
        self.top_nn = MLP(arch_top, out_activation='sigmoid')
        self.interaction_op = SecondOrderFeatureInteraction(self_interaction)

    def call(self, input):
        input_dense, input_cat = input
        emb_x = [E(x) for E, x in zip(self.emb, tf.unstack(input_cat, axis=1))]
        dense_x = self.bot_nn(input_dense)

        Z = self.interaction_op(emb_x + [dense_x])
        z = tf.concat([dense_x, Z], axis=1)
        p = self.top_nn(z)

        return p


class DLRMModel(Model):

    def __init__(self, preprocessing_settings: PreprocessorSettings,
                 model_name,
                 is_resample=False,
                 is_class_weights=False,
                 is_focal_loss=False,
                 is_label_smoothing=False,
                 is_feature_importance=False,
                 feature_importance_path='',
                 raw_train_data_path=''):
        super().__init__(preprocessing_settings=preprocessing_settings, model_name=model_name)

        self.output_folder = 'output_folder_DLRMModel'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.best_model_output_path = os.path.join(self.output_folder, self.model_name)
        self.is_resample = is_resample
        self.is_class_weight = is_class_weights
        self.raw_train_data_path = raw_train_data_path
        self.is_subsample_train = False
        self.is_focal_loss = is_focal_loss
        self.is_label_smoothing = is_label_smoothing
        self.is_feature_importance = is_feature_importance
        self.feature_importance_path = feature_importance_path

    def get_model(self):
        model = DLRM(
            embedding_sizes=self.emb_counts,
            embedding_dim=2,
            arch_bot=[8, 2],
            arch_top=[128, 64, 1],
            self_interaction=False
        )

        if self.is_focal_loss:
            loss = tfa.losses.SigmoidFocalCrossEntropy(
                from_logits=False)
        else:
            if self.is_label_smoothing:
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1)
            else:
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=loss,
            metrics=['accuracy']
        )

        return model

    def subsample_train(self, df, ratio):
        df = df.sample(frac=ratio)  # Shuffle the dataframe
        return df

    def split_data(self):
        if self.is_feature_importance:
            self.renew_features()
        # Split into 3 subset df
        self.create_train_val_test_df()

        if self.is_resample:
            self.training_df = self.resample_dataset(df=self.training_df)

        if self.is_subsample_train:
            self.training_df = self.subsample_train(df=self.training_df, ratio=0.5)

        # Create tf dataset
        self.train_ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices((
                tf.cast(self.training_df[self.numerical_features_cols].values, tf.float32),
                tf.cast(self.training_df[self.categorical_features_cols].values, tf.int32),
            )),
            tf.data.Dataset.from_tensor_slices((
                tf.cast(self.training_df['install'].values, tf.float32)
            ))
        )).shuffle(buffer_size=100)

        self.valid_ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices((
                tf.cast(self.valid_df[self.numerical_features_cols].values, tf.float32),
                tf.cast(self.valid_df[self.categorical_features_cols].values, tf.int32),
            )),
            tf.data.Dataset.from_tensor_slices((
                tf.cast(self.valid_df['install'].values, tf.float32)
            ))
        ))

        self.test_ds_x = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices((
                tf.cast(self.test_df[self.numerical_features_cols].values, tf.float32),
                tf.cast(self.test_df[self.categorical_features_cols].values, tf.int32)
            )),
        ))

        self.valid_ds_x = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices((
                tf.cast(self.valid_df[self.numerical_features_cols].values, tf.float32),
                tf.cast(self.valid_df[self.categorical_features_cols].values, tf.int32)
            )),
        ))

    def train(self):
        model = self.get_model()
        if self.is_class_weight:
            raw_train_df = pd.read_csv(self.raw_train_data_path, delimiter=';')
            neg, pos = np.bincount(raw_train_df['install'])
            total = neg + pos
            # Scaling by total/2 helps keep the loss to a similar magnitude.
            # The sum of the weights of all examples stays the same.
            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)

            class_weight = {0: weight_for_0, 1: weight_for_1}
        else:
            class_weight = None

        history = model.fit(
            self.train_ds.batch(self.batch_size),
            validation_data=self.valid_ds.batch(self.batch_size),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                # Use default monitor='val_loss'
            ],
            epochs=100,
            verbose=1,
            class_weight=class_weight
        )
        model.summary()
        results = model.evaluate(self.valid_ds.batch(self.batch_size))
        print(f'Loss {results[0]}, Accuracy {results[1]}')

        model.save_weights(self.best_model_output_path)

    def exec_eval(self, ds_x, df):
        model = self.get_model()
        model.load_weights(self.best_model_output_path)

        y_pred = model.predict(ds_x.batch(self.batch_size))
        y_pred_0 = 1 - y_pred
        y_pred = np.concatenate([y_pred_0, y_pred], axis=-1)
        y_gt = df['install'].values
        metric_dict = get_all_metrics(y_gt=y_gt, y_pred=y_pred, visualize=self.visualize, model_name=self.model_name)
        return metric_dict, y_gt, y_pred

    def eval_on_valid(self):
        metric_dict, y_gt, y_pred = self.exec_eval(ds_x=self.valid_ds_x, df=self.valid_df)
        return metric_dict, y_gt, y_pred

    def eval_on_test(self):
        metric_dict, y_gt, y_pred = self.exec_eval(ds_x=self.test_ds_x, df=self.test_df)
        return metric_dict, y_gt, y_pred

    def renew_features(self):
        if self.is_feature_importance and os.path.exists(self.feature_importance_path):
            considered_features = json.load(open(self.feature_importance_path, "r"))
            # Filter out some numerical columns
            new_numerical_features_cols = [c for c in self.numerical_features_cols if c in considered_features]
            if new_numerical_features_cols:
                self.numerical_features_cols = new_numerical_features_cols
            # Filter out some categorical columns
            new_categorical_features_cols = [(c, e) for c, e in zip(self.categorical_features_cols, self.emb_counts) if
                                             c in considered_features]
            if new_categorical_features_cols:
                self.categorical_features_cols = [t[0] for t in new_categorical_features_cols]
                self.emb_counts = [t[1] for t in new_categorical_features_cols]

    def generate_assessment_prediction(self):
        assessment_feature_df = pd.read_csv(self.preprocessing_settings.assessment_features_path, delimiter=';')[
            self.all_cols_and_id]

        if self.is_feature_importance:
            self.renew_features()

        model = self.get_model()
        model.load_weights(self.best_model_output_path)

        assessement_ds_x = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices((
                tf.cast(assessment_feature_df[self.numerical_features_cols].values, tf.float32),
                tf.cast(assessment_feature_df[self.categorical_features_cols].values, tf.int32)
            )),
        ))

        y_pred = model.predict(assessement_ds_x.batch(self.batch_size))
        assessment_feature_df_x_id = assessment_feature_df[['id']].values
        result_values = np.concatenate([assessment_feature_df_x_id, y_pred], axis=-1)
        submission_df = pd.DataFrame(result_values, columns=['id',
                                                             'install_proba'])
        # add some count
        y_pred_0 = 1 - y_pred
        processed_y_pred = np.concatenate([y_pred_0, y_pred], axis=-1)
        count_dict = return_counts_y_pred(y_pred=processed_y_pred)
        return submission_df, count_dict

    def clean_variables(self):
        self.training_df = None
        self.test_df = None
        self.valid_df = None
        self.train_ds = None
        self.valid_ds = None
        self.valid_ds_x = None
        self.test_ds_x = None

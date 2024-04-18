import numpy as  np
import pandas as pd
import tensorflow as tf

train_feature_path = '../a_preprocessing/output_folder_explore_preprocessor/train_features.csv'
assessment_feature_path = '../a_preprocessing/output_folder_explore_preprocessor/assessment_features.csv'

numerical_features_cols = ['startCount_gaussian',
                           'viewCount_gaussian',
                           'clickCount_gaussian',
                           'installCount_gaussian',
                           'startCount7d_gaussian',
                           'campaignId_install_over_frequency',
                           'sourceGameId_install_over_frequency']

train_feature_df = \
    pd.read_csv(
        train_feature_path,
        delimiter=';')[numerical_features_cols + ["install"]]


class Autoencoder(tf.keras.Model):
    def __init__(
            self,
            hidden_dim,
            output_dim
    ):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.layers.Dense(units=hidden_dim, activation=tf.nn.relu)
        self.decoder = tf.keras.layers.Dense(units=output_dim, activation=tf.nn.tanh)

    def call(self, input):
        input = self.encoder(input)
        output = self.decoder(input)
        return output


x_train = train_feature_df[numerical_features_cols].values
autoencoder = Autoencoder(hidden_dim=int(x_train.shape[-1] // 2), output_dim=x_train.shape[-1])

opt = tf.optimizers.Adam(lr=0.0001, decay=1e-5)
autoencoder.compile(loss='mse', optimizer=opt)
history = autoencoder.fit(x_train,
                          x_train,
                          batch_size=128,
                          epochs=1,
                          validation_split=0.2)

decoded = autoencoder.predict(x_train, batch_size=128)
errors = []
# loop over all original images and their corresponding
# reconstructions
for (image, recon) in zip(x_train, decoded):
    # compute the mean squared error between the ground-truth image
    # and the reconstructed image, then add it to our list of errors
    mse = np.mean((image - recon) ** 2)
    errors.append(mse)

# compute the q-th quantile of the errors which serves as our
# threshold to identify anomalies
thresh = np.quantile(errors, 0.98)
idxs = np.where(np.array(errors) >= thresh)[0]

assessment_feature_df = \
    pd.read_csv(
        assessment_feature_path,
        delimiter=';')[numerical_features_cols + ['id']]

assessment_x = assessment_feature_df[numerical_features_cols].values

decoded = autoencoder.predict(assessment_x)
errors = []
# loop over all original images and their corresponding
# reconstructions
for (image, recon) in zip(assessment_x, decoded):
    # compute the mean squared error between the ground-truth image
    # and the reconstructed image, then add it to our list of errors
    mse = np.mean((image - recon) ** 2)
    errors.append(mse)

idxs = np.where(np.array(errors) >= thresh)[0]

submission_df = assessment_feature_df[['id']].reset_index(drop=True)
submission_df['install_proba'] = 0

submission_df.loc[idxs, 'install_proba'] = 1

print("All done for autoencoder!")

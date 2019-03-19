import os
from keras.models import load_model
from evaluate import f1
from cutsom_generator import CustomGenerator
from constant import *
from IPython.display import display
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(log_info + 'Apart data mode!!!' if debug_mode else 'Full data mode!!!')


def get_test_sample():
    return list(set([test_sample[:36] for test_sample in os.listdir(test_dir)]))


submission = pd.read_csv(submission_csv)
display(submission.head(), submission.describe())

test_x = get_test_sample()
if debug_mode:
    test_x = test_x[0:test_samples]
generator_test = CustomGenerator(root_path=test_dir,
                                 sample_x=test_x,
                                 labels=None,
                                 batch_size=batch_size,
                                 augment=False,
                                 dim=(input_dim, input_dim),
                                 n_channels=input_channel,
                                 n_classes=n_classes,
                                 shuffle=False)

model = load_model(weight_dir, custom_objects={'f1': f1})
predict = model.predict_generator(generator=generator_test,
                                  steps=len(generator_test))
display(predict[:10, :])
prediction = []

for row in tqdm(range(submission.shape[0])):
    str_label = ''
    for col in range(predict.shape[1]):
        if (predict[row, col] < 0.1):
            str_label += ''
        else:
            str_label += str(col) + ' '
    prediction.append(str_label.strip())

display(prediction)
submission['Predicted'] = np.array(prediction)
submission.to_csv(predict_csv, index=False)

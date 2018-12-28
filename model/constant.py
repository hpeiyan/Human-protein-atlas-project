import os

debug_mode = False
log_info = '[log_info] '

main_dir = r'/home/bigdata/Documents/DeepLearningProject/datasets/human-protien-data'
test_dir = os.path.join(main_dir, 'test')
train_dir = os.path.join(main_dir, 'train')
info_img_dir = os.path.join(main_dir, 'config.png')
augm_img_dir = os.path.join(main_dir, 'augment.png')
weight_dir = os.path.join(main_dir, 'weights.hdf5')
submission_csv = os.path.join(main_dir, 'submission.csv')
predict_csv = os.path.join(main_dir, 'predict.csv')
score_val = os.path.join(main_dir, 'score.npy')
input_dim = 224
input_channel = 3
batch_size = 128
patience = 5
n_classes = 28
epochs = 30
THRESHOLD = 0.5
test_samples = 1000

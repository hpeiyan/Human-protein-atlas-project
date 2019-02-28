import matplotlib.pyplot as plt
import math


def plot(history, img_path):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_f1 = history.history['f1']
    val_f1 = history.history['val_f1']

    epochs = range(1, len(val_f1) + 1)

    plt.plot(epochs, train_f1, label='train_f1')
    plt.plot(epochs, val_f1, label='val_f1')
    plt.legend()

    plt.figure()
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.legend()
    plt.savefig(img_path)


def plot_img(imgs, img_path):
    fig, axes = plt.subplots(1, 8, figsize=(50, 50))

    for idx in range(8):
        axes[idx].imshow(imgs[idx,])
    plt.savefig(img_path)
    # plt.show()

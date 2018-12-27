import matplotlib.pyplot as plt


def plot(history, img_path):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_f1 = history.history['f1']
    val_f1 = history.history['val_f1']

    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, label='train_acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.legend()
    plt.figure()
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.legend()
    plt.figure()
    plt.plot(epochs, train_f1, label='train_f1')
    plt.plot(epochs, val_f1, label='val_f1')
    plt.legend()
    plt.savefig(img_path)

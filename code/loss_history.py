import keras
import matplotlib.pyplot as pyplot


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        pyplot.figure()
        # acc
        pyplot.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        pyplot.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            pyplot.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            pyplot.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        pyplot.grid(True)
        pyplot.xlabel(loss_type)
        pyplot.ylabel('acc-loss')
        pyplot.legend(loc="upper right")
        pyplot.show()
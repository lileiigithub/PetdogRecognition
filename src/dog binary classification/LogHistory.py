import keras
from datetime import datetime 

class LogHistory(keras.callbacks.Callback):
    def __init__(self,_path):
        self.path = _path
        with open(self.path,'a') as f:
            f.write('\nnew start:\n')
            
    def on_train_begin(self, logs={}):
        pass
# 系统已定义好的接口:on_batch_end
    def on_batch_end(self, batch, logs={}):
        now_loss = float(logs.get('loss'))
        now_acc = float(logs.get('acc'))
#        print(type(now_loss),type(now_acc))
#        print(now_loss,now_acc)
        with open(self.path,'a') as f:
            f.write(str(datetime.now().isoformat(' '))+'   ')
            f.write(
                    'loss: '+'{:0<6}'.format(round(now_loss,4))+'   '
                    'acc: '+'{:0<6}'.format(round(now_acc,4))+'\n'
                    )

    def on_epoch_end(self, epoch, logs=None):
        now_val_loss = logs.get('val_loss')
        now_val_acc = logs.get('val_acc')
        with open(self.path,'a') as f:
            f.write(str(datetime.now().isoformat(' '))+'   ')
            f.write(
                    'val_loss: '+str(now_val_loss)+'   '  #'{: <8}'.format(now_val_loss)
                    'val_acc: '+str(now_val_acc)+'\n'  # '{: <6}'.format(now_val_acc)
                    )

            
#plt.plot()
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
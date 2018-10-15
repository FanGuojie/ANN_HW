import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')


def show(loss,acc,max_epoch,test_epoch):
    font = {'weight': 'bold','size': 10}
    plt.figure('cnn_mnist')
    plt.rc('font',**font)
    plt.subplot(2,1,1)
    plt.title('loss',fontsize=14)
    idx=np.arange(1,max_epoch+1,test_epoch)
    plt.plot(idx,loss,'bo-')
    plt.xlabel('epoch',fontsize=12)
    plt.ylabel('test loss',fontsize=12)
    plt.grid()

    plt.subplot(2,1,2)
    plt.title("accuracy",fontsize=14)
    idx = np.arange(1, max_epoch + 1, test_epoch)
    plt.plot(idx, acc, 'ro-')
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('test accuracy', fontsize=12)
    plt.grid()

    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    plt.suptitle("cnn_mnist", fontsize=18, x=0.55, y=1)
    plt.savefig("cnn_mnist.png")

    plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_stats(hist):
    n = len(hist['train']['loss'])
    n_ = len(hist['test']['loss'])
    
    loss = hist['train']['loss']
    acc = hist['train']['accuracy']
    val_loss = hist['test']['loss']
    val_acc = hist['test']['accuracy']

    loss_patch_train = mpatches.Patch(color='red', label='Training Loss')
    loss_patch_dev = mpatches.Patch(color='blue', label='Dev Loss')
    acc_patch_train = mpatches.Patch(color='red', label='Training Accuracy')
    acc_patch_dev = mpatches.Patch(color='blue', label='Dev Accuracy')

    fig = plt.figure(0)
    plt.suptitle('Train Statistics')

    ax1 = fig.add_subplot(121)
    ax1.set_title('Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.plot(list(range(n)), loss, color='red')
    ax1.plot(list(range(n)), val_loss, color='blue')
    ax1.legend(handles=[loss_patch_train, loss_patch_dev])

    ax2 = fig.add_subplot(122)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.plot(list(range(n)), acc, color='red')
    ax2.plot(list(range(n)), val_acc, color='blue')
    ax2.legend(handles=[acc_patch_train, acc_patch_dev])

    plt.tight_layout()
    plt.savefig('result.png')
    plt.show()
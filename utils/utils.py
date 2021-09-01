class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def updateLR(max_lr, epoch, total_epoch):
    # decay = 1.0
    # # if epoch >= 10:
    # #     decay = 0.8
    # # elif epoch >= 30:
    # #     decay = 0.8
    #
    # # lr = lr_init * decay
    # # if lr <= 0.000000001:
    # #     lr = 0.000000001
    # if epoch<2:
    #     lr=0.005
    # elif epoch<5:
    #     lr=0.001
    # elif epoch<8:
    #     lr=0.0005
    # elif epoch<10:
    #     lr=0.0001
    # elif epoch<15:
    #     lr=0.00005
    # else:
    #     lr=0.00001
    # return lr

    # decay = 0.8
    # if epoch<1:
    #     lr=0.005
    # elif epoch<3:
    #     lr=0.001
    # elif epoch<5:
    #     lr=0.0005
    # # elif epoch<10:
    # #     lr=0.0001
    # # elif epoch<15:
    # #     lr=0.00005
    # else:
    #     lr=lr_init*decay
    #
    # if lr <= 0.00000001:
    #     lr = 0.00000001

    """
    Implements gradual warmup, if train_steps < warmup_steps, the
    learning rate will be `train_steps/warmup_steps * init_lr`.
    Args:
        warmup_steps:warmup步长阈值,即train_steps<warmup_steps,使用预热学习率,否则使用预设值学习率
        train_steps:训练了的步长数
        init_lr:预设置学习率
    """
    warmup_steps = 20
    lr_0 = 1e-6
    # max_lr = 0.003
    end_lr = 1e-7
    # total_epoch = 330
    warmup_learning_rate = max_lr
    sigma = 0.98

    if epoch < warmup_steps:
        lr_step = (max_lr - lr_0) / warmup_steps
        warmup_learning_rate = lr_step * epoch  # gradual warmup_lr
        lr = warmup_learning_rate
    else:
        # learning_rate = np.sin(learning_rate)  #预热学习率结束后,学习率呈sin衰减
        lr_step = (max_lr - end_lr) / (total_epoch - warmup_steps)
        lr = warmup_learning_rate - ((epoch - warmup_steps) * lr_step) ** sigma # 预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)  0.85
        # lr=max_lr*(0.98**(epoch-warmup_steps))
        if lr <= end_lr:
            lr = end_lr
    return lr

if __name__ == '__main__':
    max_lr=0.002
    total_epoch=300
    for epoch in range(total_epoch):
        print(epoch,"{:.9f}".format(updateLR(max_lr, epoch, total_epoch)))
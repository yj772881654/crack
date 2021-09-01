def adjust_lr(init_lr,now_it,total_it):
    power = 0.9
    lr = init_lr * (1 - float(now_it) / total_it) ** power
    return lr
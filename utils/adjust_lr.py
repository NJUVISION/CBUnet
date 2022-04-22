def scaled_learning_rate(optimizer, init_lr, epoch=0, drop_epoch=16, rate=2.0):
    if epoch < 60:
        return
    lr = init_lr / rate ** (epoch // drop_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

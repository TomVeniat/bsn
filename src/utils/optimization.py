def adjust_lr(optimizer, epoch, tresh, val, logger=None, except_groups=None):
    if except_groups is None:
        except_groups = []
    assert len(tresh) == len(val) - 1
    i = 0
    while i < len(tresh) and epoch > tresh[i]:
        i += 1
    lr = val[i]

    if logger is not None:
        logger.info('Setting learning rate to {:.5f} (except for following {})'.format(lr, len(except_groups)))

    for param_group in optimizer.param_groups:
        if param_group['name'] not in except_groups:
            param_group['lr'] = lr
        elif logger is not None:
            logger.info('{} - {}'.format(param_group['name'], param_group['lr']))

    return lr

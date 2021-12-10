import torch


def set_device_from_args(args):
    # choose device
    if torch.cuda.is_available() and not args.no_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    setattr(args, 'device', device)

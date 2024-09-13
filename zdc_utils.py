import torch
import numpy as np


def sum_channels_parallel_numpy(data):
    coords = np.ogrid[0:data.shape[1], 0:data.shape[2]]
    half_x = data.shape[1] // 2
    half_y = data.shape[2] // 2

    checkerboard = (coords[0] + coords[1]) % 2 != 0
    checkerboard = checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])

    ch5 = (data * checkerboard).sum(axis=1).sum(axis=1)

    checkerboard = (coords[0] + coords[1]) % 2 == 0
    checkerboard = checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, :half_x, :half_y] = checkerboard[:, :half_x, :half_y]
    ch1 = (data * mask).sum(axis=1).sum(axis=1)

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, :half_x, half_y:] = checkerboard[:, :half_x, half_y:]
    ch2 = (data * mask).sum(axis=1).sum(axis=1)

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, half_x:, :half_y] = checkerboard[:, half_x:, :half_y]
    ch3 = (data * mask).sum(axis=1).sum(axis=1)

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, half_x:, half_y:] = checkerboard[:, half_x:, half_y:]
    ch4 = (data * mask).sum(axis=1).sum(axis=1)

    # assert all(ch1+ch2+ch3+ch4+ch5 == data.sum(axis=1).sum(axis=1))==True

    return zip(ch1, ch2, ch3, ch4, ch5)


def sum_channels_parallel_pytorch(data, args):
    coords = torch.arange(data.shape[1]).reshape(-1, 1).to(args["device"]), \
             torch.arange(data.shape[2]).reshape(1, -1).to(args["device"])
    half_x = data.shape[1] // 2
    half_y = data.shape[2] // 2

    checkerboard = (coords[0] + coords[1]) % 2 != 0
    checkerboard = checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])

    ch5 = (data * checkerboard).sum(axis=1).sum(axis=1)

    checkerboard = (coords[0] + coords[1]) % 2 == 0
    checkerboard = checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])

    mask = torch.zeros((1, data.shape[1], data.shape[2])).to(args["device"])
    mask[:, :half_x, :half_y] = checkerboard[:, :half_x, :half_y]
    ch1 = (data * mask).sum(axis=1).sum(axis=1)

    mask = torch.zeros((1, data.shape[1], data.shape[2])).to(args["device"])
    mask[:, :half_x, half_y:] = checkerboard[:, :half_x, half_y:]
    ch2 = (data * mask).sum(axis=1).sum(axis=1)

    mask = torch.zeros((1, data.shape[1], data.shape[2])).to(args["device"])
    mask[:, half_x:, :half_y] = checkerboard[:, half_x:, :half_y]
    ch3 = (data * mask).sum(axis=1).sum(axis=1)

    mask = torch.zeros((1, data.shape[1], data.shape[2])).to(args["device"])
    mask[:, half_x:, half_y:] = checkerboard[:, half_x:, half_y:]
    ch4 = (data * mask).sum(axis=1).sum(axis=1)

    # assert all(ch1+ch2+ch3+ch4+ch5 == data.sum(axis=1).sum(axis=1))==True
    #print(ch1.item(), ch2.item(), ch3.item(), ch4.item(), ch5.item())

    #return zip(ch1.item(), ch2.item(), ch3.item(), ch4.item(), ch5.item())
    return zip(ch1, ch2, ch3, ch4, ch5)
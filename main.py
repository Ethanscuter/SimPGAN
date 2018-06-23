import os
from torch.backends import cudnn
# from data_loader import get_loader, get_sample_loader
from data_loader_pair import get_loader, get_sample_loader
# from solver import Solver
# from solver_reid import Solver
# from solver_siamese import Solver
from solver_double_siamese_im1 import Solver
# from solver_double_siamese_im2 import Solver


def main():
    market_loader, grid_loader = get_loader(image_size, market_path, grid_path, market_list, grid_list, batch_size,
                                            num_workers)

    sample_probe_loader = get_sample_loader('./grid/probe')
    sample_gallery_loader = get_sample_loader('./grid/gallery')

    solver = Solver(use_reconst_loss, use_labels, num_classes_market, num_classes_grid, beta1, beta2, g_conv_dim,
                    d_conv_dim, train_iters, batch_size, lr, log_step, sample_step, sample_path, model_path,
                    market_loader, grid_loader, sample_probe_loader, sample_gallery_loader)
    cudnn.benchmark = True

    # create directories if not exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    if mode == 'train':
        solver.train()
    elif mode == 'sample':
        solver.sample()


if __name__ == '__main__':
    # model hyper-parameters
    image_size = 224
    g_conv_dim = 64
    d_conv_dim = 64
    use_reconst_loss = True
    use_labels = False
    num_classes_market = 751
    num_classes_grid = 250

    # training hyper-parameters
    train_iters = 130000
    batch_size = 1
    num_workers = 0
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    # misc
    mode = 'train'
    model_path = './models'
    sample_path = './samples'
    market_path = './market/train'
    market_list = './market/market_train.list'
    grid_path = './grid/train'
    grid_list = './grid/grid_train.list'
    log_step = 10
    sample_step = 10

    main()

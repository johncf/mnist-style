from argparse import ArgumentParser


def cli_parser_add_arguments(parser: ArgumentParser, batch_size=64, epochs=12, lr=4e-4,
                             feat_size=8, ckpt_dir='./pt-aae', data_dir='./data'):
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='B',
                        help=f'batch size for training and testing (default: {batch_size})')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='E',
                        help=f'number of epochs to train (default: {epochs})')
    parser.add_argument('--lr', type=float, default=lr,
                        help=f'learning rate with adam optimizer (default: {lr})')
    parser.add_argument('--feat-size', type=int, default=feat_size, metavar='N',
                        help=f'dimensions of the latent feature vector (default: {feat_size})')
    parser.add_argument('--ckpt-dir', default=ckpt_dir, metavar='ckpt',
                        help=f'training session directory (default: {ckpt_dir}) ' +
                             'for storing model parameters and trainer states')
    parser.add_argument('--data-dir', default=data_dir, metavar='data',
                        help=f'MNIST data directory (default: {data_dir}) ' +
                             '(gets created and downloaded to, if doesn\'t exist)')

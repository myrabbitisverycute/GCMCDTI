import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", help='train')
    parser.add_argument('--data_type', type=str, default="DPdata")
    parser.add_argument('--model-path', type=str, default="./models")
    parser.add_argument('--data-path', type=str, default="./data")
    parser.add_argument('--data-shuffle', type=bool, default=False)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--val-step', type=int, default=1)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--neg-cnt', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--slope', type=float, default=0.3)
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--emb-dim', type=int, default=32)
    parser.add_argument('--hidden', default=[64, 32, 16, 8])
    parser.add_argument('--nb', type=int, default=1)
    args = parser.parse_args()

    return args

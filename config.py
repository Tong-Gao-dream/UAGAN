import argparse

def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./dataset/img', help='input RGB image path')
    parser.add_argument('--mask_dir', type=str, default='./dataset/mask', help='input mask path')
    parser.add_argument('--lr', type=float, default='0.0002', help='learning rate')
    parser.add_argument('--batch_size', type=int, default='5', help='batch_size in training')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--epoch", type=int, default=600, help="epoch in training")

    args = parser.parse_args()
    return args
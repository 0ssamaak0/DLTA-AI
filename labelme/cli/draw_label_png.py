import argparse

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from labelme.logger import logger
from labelme import utils


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('label_png', help='label PNG file')
    args = parser.parse_args()

    lbl = np.asarray(PIL.Image.open(args.label_png))

    logger.info('label shape: {}'.format(lbl.shape))
    logger.info('unique label values: {}'.format(np.unique(lbl)))

    lbl_viz = utils.draw_label(lbl)
    plt.imshow(lbl_viz)
    plt.show()


if __name__ == '__main__':
    main()

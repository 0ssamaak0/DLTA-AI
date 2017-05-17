import json
import os.path as osp

import nose
import numpy as np

from labelme import utils


here = osp.dirname(osp.abspath(__file__))


def test_img_b64_to_array():
    json_file = osp.join(here, '../_static/IMG_6319.json')
    data = json.load(open(json_file))
    img_b64 = data['imageData']
    img = utils.img_b64_to_array(img_b64)
    nose.tools.assert_equal(img.dtype, np.uint8)
    nose.tools.assert_equal(img.shape, (3024, 4032, 3))

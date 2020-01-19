from config import get_arguments
import numpy as np
import SinGAN.functions as functions
import cv2
import os


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Inpainting')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='inpainting_telea')
    opt = parser.parse_args()

    opt = functions.post_config(opt)
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        mask = cv2.imread('%s/%s' % (opt.ref_dir, opt.ref_name),0)
        img = cv2.imread('%s/%s' % (opt.input_dir, opt.input_name))
        telea_inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        #res = cv2.bitwise_and(dst, dst, mask=mask)

        cv2.imwrite('%s/%s_telea%s' % (dir2save,opt.input_name[:-4],opt.input_name[-4:]),telea_inpainted)

        cv2.destroyAllWindows()






if __name__ == '__main__':

    import argparse
    import os
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--left', dest='left_data_folder',
                        help='path to left result')

    parser.add_argument('--right', dest='right_data_folder',
                        help='path to left result')

    parser.add_argument('--res_dir', dest='res_dir',
                        help='path to dir with compare result')

    args = parser.parse_args()

    res_dir = os.path.join('results', args.res_dir)
    os.makedirs(res_dir, exist_ok=True)

    left_dir_name = os.path.join(args.left_data_folder, 'images')
    right_dir_name = os.path.join(args.right_data_folder, 'images')

    im_names = os.listdir(left_dir_name)
    ends = ['fake', 'real', 'rec', 'A_img.png', 'B_img.png', 'A_seg.png', 'B_seg.png']
    ends = [f"_{x}" for x in ends]
    for i, _ in enumerate(im_names):
        for end in ends:
            im_names[i] = im_names[i].replace(end, '')

    im_names = sorted(list(set(im_names)))

    res = None
    for im_name in im_names:
        source = cv2.imread(os.path.join(left_dir_name, f'{im_name}_real_A_img.png'))
        fake_left = cv2.imread(os.path.join(left_dir_name, f'{im_name}_fake_B_img.png'))
        fake_right = cv2.imread(os.path.join(right_dir_name, f'{im_name}_fake_B_img.png'))
        to_curly = np.hstack((source, fake_left, fake_right))
        source = cv2.imread(os.path.join(left_dir_name, f'{im_name}_real_B_img.png'))
        fake_left = cv2.imread(os.path.join(left_dir_name, f'{im_name}_fake_A_img.png'))
        fake_right = cv2.imread(os.path.join(right_dir_name, f'{im_name}_fake_A_img.png'))
        to_straight = np.hstack((source, fake_left, fake_right))
        line = np.hstack((to_curly, to_straight))
        cv2.imwrite(os.path.join(res_dir, f'{im_name}.png'), line)
        if res is None:
            res = line
        else:
            res = np.vstack((res, line))
    cv2.imwrite(os.path.join(res_dir, f'{args.res_dir}.png'), res)

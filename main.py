# This is a tool for scaling up a greyscale image with pixel-averaging technique
# Developed by AMIR RASTAR

import func_binary

if __name__ == '__main__':
    denoise = False
    alg = 3   # 1: random forest, 2: K nearest neighbours'
    if alg == 1:
        num = 25  # number of trees in the ensemble
    elif alg == 2:
        num = 50  # number of neighbours
    elif alg == 3:
        num = 4  # Decision trees
    else:
        num = 0
        exit()

    grid_search = False  # Takes longer if true
    image_path = 'ct.jpg'
    num_zooms = 2
    func_binary.zoomx(num_zooms, image_path, alg, num, grid_search, denoise)

import funcs

if __name__ == '__main__':
    alg = 1   # 1: random forest, 2: K nearest neighbours'
    if alg == 1:
        num = 25  # number of trees in the ensemble
    elif alg == 2:
        num = 50  # number of neighbours
    else:
        num = 0
        exit()

    grid_search = False  # Takes longer if true
    image_path = 'ax2.jpg'
    num_zooms = 5
    funcs.zoomx(num_zooms, image_path, alg, num, grid_search)

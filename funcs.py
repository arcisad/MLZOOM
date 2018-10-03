from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from PIL import Image
import math
import pyprind
import os


def reduce_image(image):
    def get_data_from_reduced(my_list):
        in_list = []
        out_list = []
        r_c = my_list.shape[3]
        c_c = my_list.shape[2]
        progbar2 = pyprind.ProgBar(my_list.shape[0] * my_list.shape[1],
                                   title='Extracting data from reduced part ...')
        for kk in range(my_list.shape[0]):
            for ll in range(my_list.shape[1]):
                ii = 0
                while ii + 1 < c_c:
                    jj = 0
                    while jj + 1 < r_c:
                        block = np.zeros((2, 2, 3))
                        block_list = []
                        for nn in [0, 1]:
                            for mm in [0, 1]:
                                for zz in [0, 1, 2]:
                                    block[nn, mm, zz] = my_list[kk, ll, ii + nn, jj + mm, zz]
                                block_list.extend(list(my_list[kk, ll, ii + nn, jj + mm, :]))
                        out_list.append(block_list)
                        mean = np.mean(np.mean(block, 1), 0)
                        in_list.append(mean)
                        jj += 2
                    ii += 2

                progbar2.update()
        return in_list, out_list

    im = Image.open(image, 'r')
    img = im.load()
    siz = im.size
    w, h = calculate_aspect(siz[0], siz[1])
    if h / w != 1:
        print('Image aspect ratio is %d:%d ... ' % (w, h))
        if h > 20 and w > 20:
            print('Please use a standard format aspect ratio ... \n Exiting ...')
            exit()
        new_w = int(siz[0] / w)
        new_h = int(siz[1] / h)
        sixteen_list = np.zeros((new_w, new_h, w, h, 3))
        new_img = np.zeros((new_w, new_h, 3))
        s_r_co = 0
        s_c_co = 0
        progbar = pyprind.ProgBar(siz[0] * siz[1], title='Reducing image...')
        for i in range(siz[0]):
            for j in range(siz[1]):
                pixel = img[i, j]
                c_counter = math.floor(i / w)
                r_counter = math.floor(j / h)
                n_pixel = np.array(pixel)
                for k in [0, 1, 2]:
                    new_img[c_counter, r_counter, k] += n_pixel[k]
                    sixteen_list[c_counter, r_counter, s_c_co, s_r_co, k] = n_pixel[k]
                progbar.update()
                if s_r_co == h - 1:
                    s_r_co = 0
                else:
                    s_r_co += 1
            if s_c_co == w - 1:
                s_c_co = 0
            else:
                s_c_co += 1
        new_img = (new_img / (w * h)) / 255
        new_sixteen = sixteen_list / 255
        input_list, output_list = get_data_from_reduced(new_sixteen)
    else:
        new_img = np.zeros((siz[0], siz[1], 3))
        for i in range(siz[0]):
            for j in range(siz[1]):
                pixel = img[i, j]
                n_pixel = np.array(pixel) / 255
                for k in [0, 1, 2]:
                    new_img[i, j, k] = n_pixel[k]
        input_list = []
        output_list = []

    print('Finalizing data extraction ...')
    final_image, final_input, final_output = setup_data(new_img, input_list, output_list)
    print('Data extraction finished ...')

    return final_image, final_input, final_output


def setup_data(image, inplist, outlist):
    fin_inp = inplist
    fin_out = outlist
    c_c = image.shape[0]
    r_c = image.shape[1]
    new_img = image
    while r_c * c_c > 1:
        ii = 0
        jj = 0
        while ii + 1 < c_c:
            jj = 0
            while jj + 1 < r_c:
                block = np.zeros((2, 2, 3))
                block_list = []
                for nn in [0, 1]:
                    for mm in [0, 1]:
                        for zz in [0, 1, 2]:
                            block[nn, mm, zz] = image[ii + nn, jj + mm, zz]
                        block_list.extend(list(image[ii + nn, jj + mm, :]))
                fin_out.append(block_list)
                mean = np.mean(np.mean(block, 1), 0)
                fin_inp.append(mean)
                jj += 2
            ii += 2

        r_c = int(jj / 2)
        c_c = int(ii / 2)
        new_img = np.zeros((c_c, r_c, 3))
        for i in range(ii):
            for j in range(jj):
                pixel = image[i, j, :]
                c_counter = math.floor(i / 2)
                r_counter = math.floor(j / 2)
                n_pixel = np.array(pixel)
                for k in [0, 1, 2]:
                    new_img[c_counter, r_counter, k] += n_pixel[k]
        new_img = new_img / 4
        image = new_img
    return new_img, np.array(fin_inp), np.array(fin_out)


# https://gist.github.com/Integralist
def calculate_aspect(width: int, height: int):
    temp = 0

    def gcd(a, b):
        """The GCD (greatest common divisor) is the highest number that evenly divides both width and height."""
        return a if b == 0 else gcd(b, a % b)

    if width == height:
        return 1, 1

    if width < height:
        temp = width
        width = height
        height = temp

    divisor = gcd(width, height)
    print(divisor)

    x = int(width / divisor) if not temp else int(height / divisor)
    y = int(height / divisor) if not temp else int(width / divisor)

    return x, y


def rmse(predictions, targets):
    def_mean = predictions.shape[0]
    sum_errs = np.sum((predictions - targets) ** 2)
    return np.sqrt(sum_errs / def_mean)


def zoomx(x_times, image, alg, nums, gsearch):
    _, fin_inp, fin_out = reduce_image(image)

    if alg == 1:
        rfc_rgs = RandomForestRegressor(n_estimators=nums, n_jobs=-1)
        if gsearch:
            param_grid = [{'n_estimators': [10, 25, 50, 75, 100, 150]}]
            grid_search = GridSearchCV(rfc_rgs, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(fin_inp, fin_out)
            n_grid = grid_search.best_params_['n_estimators']
            print("Best number of trees: %d " % n_grid)
            rfc_rgs = RandomForestRegressor(n_estimators=n_grid, n_jobs=-1)
        rfc_rgs.fit(fin_inp, fin_out)
    else:
        knn_rgs = KNeighborsRegressor(weights='distance', n_jobs=-1, n_neighbors=nums)
        if gsearch:
            param_grid = [{'n_neighbors': [10, 25, 50, 75, 100, 150]}]
            grid_search = GridSearchCV(knn_rgs, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(fin_inp, fin_out)
            n_grid = grid_search.best_params_['n_neighbors']
            print("Best number of neighbours: %d " % n_grid)
            knn_rgs = KNeighborsRegressor(weights='distance', n_jobs=-1, n_neighbors=n_grid)
        knn_rgs.fit(fin_inp, fin_out)

    im = Image.open(image, 'r')
    img = im.load()
    out_image = img
    siz = np.array(im.size)
    in_image = np.zeros((siz[0], siz[1], 3))
    for ii in range(siz[0]):
        for jj in range(siz[1]):
            pixel = np.array(out_image[ii, jj])
            for zz in [0, 1, 2]:
                in_image[ii, jj, zz] = pixel[zz] / 255
    out_image = in_image
    for i in range(x_times - 1):
        in_list = []
        for ii in range(siz[0]):
            for jj in range(siz[1]):
                pixel = out_image[ii, jj, :]
                in_list.append(list(pixel))
        print(np.array(in_list).shape)

        if alg == 1:
            pred = rfc_rgs.predict(np.array(in_list))
            print('Random forest score is: ' + str(rfc_rgs.score(fin_inp, fin_out)))
        else:
            pred = knn_rgs.predict(np.array(in_list))
            print('KNN score is: ' + str(knn_rgs.score(fin_inp, fin_out)))

        siz[0] = 2 * siz[0]
        siz[1] = 2 * siz[1]
        out_image = np.zeros((siz[0], siz[1], 3))
        p_counter = 0
        ii = 0
        while ii + 1 < siz[0]:
            jj = 0
            while jj + 1 < siz[1]:
                c_counter = 0
                for nn in [0, 1]:
                    for mm in [0, 1]:
                        for zz in [0, 1, 2]:
                            out_image[ii + nn, jj + mm, zz] = pred[p_counter, c_counter]
                            c_counter += 1
                p_counter += 1
                jj += 2
            ii += 2
    final_image = np.zeros((siz[1], siz[0], 3))
    for ii in range(siz[0]):
        for jj in range(siz[1]):
            for z in [0, 1, 2]:
                final_image[jj, ii, z] = out_image[ii, jj, z]
    final_image = Image.fromarray((final_image * 255).astype(np.uint8))
    if os.path.exists('output.jpg'):
        os.remove('output.jpg')
    final_image.save('output.jpg')
    print('Image conversion finished. Find "output.jpg" in the working directory.')
    return
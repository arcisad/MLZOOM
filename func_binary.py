from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from PIL import Image
import math
import pyprind
import os
import pandas as pd


# http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html
def convolve2d(image, kernel):
    # This function which takes an image and a kernel
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).

    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output = np.zeros_like(image)  # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.size[1] + 2, image.size[0] + 2))
    image_padded[1:-1, 1:-1] = image
    prog = pyprind.ProgBar(image.sizw[0] * image.size[1], title='Applying blur filter...')
    for x in range(image.size[0]):  # Loop over every pixel of the image
        for y in range(image.size[1]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y:y + 3, x:x + 3]).sum()
            prog.update()
    return output


def reduce_image(image, siz):
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
                        block = np.zeros((2, 2))
                        block_list = []
                        for nn in [0, 1]:
                            for mm in [0, 1]:
                                block[nn, mm] = my_list[kk, ll, ii + nn, jj + mm]
                                block_list.append(block[nn, mm])
                        out_list.append(block_list)
                        mean = np.mean(np.mean(block, 1), 0)
                        in_list.append(mean)
                        jj += 2
                    ii += 2

                progbar2.update()
        return in_list, out_list

    img = image
    w, h = calculate_aspect(siz[0], siz[1])
    if h / w != 1:
        print('Image aspect ratio is %d:%d ... ' % (w, h))
        if h > 20 and w > 20:
            print('Please use a standard format aspect ratio ... \n Exiting ...')
            exit()
        new_w = int(siz[0] / w)
        new_h = int(siz[1] / h)
        sixteen_list = np.zeros((new_w, new_h, w, h))
        new_img = np.zeros((new_w, new_h))
        s_r_co = 0
        s_c_co = 0
        progbar = pyprind.ProgBar(siz[0] * siz[1], title='Reducing image...')
        for i in range(siz[0]):
            for j in range(siz[1]):
                pixel = img[i, j]
                c_counter = math.floor(i / w)
                r_counter = math.floor(j / h)
                n_pixel = np.array(pixel)
                new_img[c_counter, r_counter] += n_pixel
                sixteen_list[c_counter, r_counter, s_c_co, s_r_co] = n_pixel
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
        new_img = np.zeros((siz[0], siz[1]))
        for i in range(siz[0]):
            for j in range(siz[1]):
                pixel = img[i, j]
                n_pixel = np.array(pixel) / 255
                new_img[i, j] = n_pixel
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
                block = np.zeros((2, 2))
                block_list = []
                for nn in [0, 1]:
                    for mm in [0, 1]:
                        block[nn, mm] = image[ii + nn, jj + mm]
                        block_list.extend([block[nn, mm]])
                fin_out.append(block_list)
                mean = np.mean(np.mean(block, 1), 0)
                fin_inp.append(mean)
                jj += 2
            ii += 2

        r_c = int(jj / 2)
        c_c = int(ii / 2)
        new_img = np.zeros((c_c, r_c))
        for i in range(ii):
            for j in range(jj):
                pixel = image[i, j]
                c_counter = math.floor(i / 2)
                r_counter = math.floor(j / 2)
                n_pixel = np.array(pixel)
                new_img[c_counter, r_counter] += n_pixel
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


def zoomx(x_times, image, alg, nums, gsearch, nr):

    im = Image.open(image, 'r')
    img = im.load()
    if not isinstance(img[0, 0], int):
        im = im.convert('L')
        img = im.load()

    _, fin_inp, fin_out = reduce_image(img, im.size)

    X_train = pd.DataFrame(fin_inp, columns={'I'})
    y_train = pd.DataFrame(fin_out)
    y_train = y_train.rename(columns={0: 'I1', 1: 'I2', 2: 'I3', 3: 'I4'})

    X_train = np.array(X_train).reshape(-1, 1)
    y_train = np.array(y_train)

    if alg == 1:
        rgs = RandomForestRegressor(n_estimators=nums, n_jobs=-1)
        if gsearch:
            param_grid = [{'n_estimators': [10, 25, 50, 75, 100, 150]}]
            grid_search = GridSearchCV(rgs, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(fin_inp, fin_out)
            n_grid = grid_search.best_params_['n_estimators']
            print("Best number of trees: %d " % n_grid)
        rgs.fit(X_train, y_train)
    elif alg == 2:
        rgs = KNeighborsRegressor(weights='distance', n_jobs=-1, n_neighbors=nums)
        if gsearch:
            param_grid = [{'n_neighbors': [10, 25, 50, 75, 100, 150]}]
            grid_search = GridSearchCV(rgs, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(fin_inp, fin_out)
            n_grid = grid_search.best_params_['n_neighbors']
            print("Best number of neighbours: %d " % n_grid)
        rgs.fit(X_train, y_train)
    elif alg == 3:
        rgs = DecisionTreeRegressor()
        rgs.fit(X_train, y_train)

    out_image = img
    siz = np.array(im.size)
    in_image = np.zeros((siz[0], siz[1]))
    for ii in range(siz[0]):
        for jj in range(siz[1]):
            pixel = np.array(out_image[ii, jj])
            in_image[ii, jj] = pixel / 255
    out_image = in_image
    for i in range(x_times - 1):
        in_list = []
        for ii in range(siz[0]):
            for jj in range(siz[1]):
                pixel = out_image[ii, jj]
                in_list.append([pixel])
        print(np.array(in_list).shape)

        X_test = pd.DataFrame(in_list, columns={'I'})
        X_test = np.array(X_test).reshape(-1, 1)

        pred = rgs.predict(X_test)
        print('Regressor score is: ' + str(rgs.score(X_train, y_train)))

        siz[0] = 2 * siz[0]
        siz[1] = 2 * siz[1]
        out_image = np.zeros((siz[0], siz[1]))
        p_counter = 0
        ii = 0
        while ii + 1 < siz[0]:
            jj = 0
            while jj + 1 < siz[1]:
                c_counter = 0
                for nn in [0, 1]:
                    for mm in [0, 1]:
                        out_image[ii + nn, jj + mm] = pred[p_counter, c_counter]
                        c_counter += 1
                p_counter += 1
                jj += 2
            ii += 2

    final_image = np.zeros((siz[1], siz[0]))
    for ii in range(siz[0]):
        for jj in range(siz[1]):
            final_image[jj, ii] = out_image[ii, jj]
    final_image = Image.fromarray((final_image * 255).astype(np.uint8))
    if nr:
        kernel = (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        final_image = Image.fromarray(convolve2d(final_image, kernel))
        final_image = Image.fromarray(convolve2d(final_image, kernel))
    if os.path.exists('output.jpg'):
        os.remove('output.jpg')
    final_image.save('output.jpg')
    print('Image conversion finished. Find "output.jpg" in the working directory.')
    return

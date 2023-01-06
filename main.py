import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread


CORR_THR = 0.0115
POSITION_TO_OFFSET_MAP = { 1: (0,0), 2: (0, -1), 3: (1,0), 4: (-1, -1) }


def imshow_img(image, out_fname='mosaic.png'):
    fig, axis = plt.subplots(1, 1, figsize=(16,6))
    axis.imshow(image, cmap='gray')
    axis.axis('off')
    # out_file_name = f"./output/part1/{file_name}.png"
    # fig.savefig(out_file_name)
    current_working_directory = os.getcwd()
    output_dir = current_working_directory + '/output/'
    fig.savefig(output_dir + out_fname)


def imshow_all(images, title=None, subtitles=None, out_fname="1234.png"):
    if subtitles is None:
        subtitles = [''] * len(images)
    ncols = len(images)
    height = 10
    width = height * len(images)
    fig, axes = plt.subplots(nrows=1, ncols=ncols,
                             figsize=(width, height))
    if title is not None:
        fig.suptitle(title, fontsize=10)
    for ax, img, label in zip(axes.ravel(), images, subtitles):
        ax.imshow(img, cmap='gray')
        ax.set_title(label, fontdict={'fontsize': 14})
        ax.axis('off')
    current_working_directory = os.getcwd()
    output_dir = current_working_directory + '/output/'
    fig.savefig(output_dir + out_fname)

def gaussian_lp_filter(image, d_0 = 50):
    m, n = image.shape
    H = np.zeros((m,n), dtype=np.float32)
    D_0 = d_0
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i - m/2)**2 + (j-n/2)**2)
            H[i,j] = np.exp(-D**2/(2*(D_0**2)))
    return H

def ideal_lp_filter(image, d_0 = 50):
    m, n = image.shape
    H = np.zeros((m,n), dtype=np.float32)
    D_0 = d_0
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i - m/2)**2 + (j-n/2)**2)
            H[i, j] = 1 if D <= D_0 else 0
    return H


def get_fft_images():
    current_working_directory = os.getcwd()
    path_to_images = current_working_directory + '/images/'
    file_names = os.listdir(path_to_images)
    images = []
    for file in file_names:
        images.append(imread( path_to_images + file, as_gray=True)) 
    return images

def plot_image_power_spectrum(images):
    for i, image in enumerate(images):
        fft = np.fft.fft2(image)
        zero_shifted = np.fft.fftshift(fft) 
        magnitude = np.abs(zero_shifted)
        power_spectrum = np.log1p(magnitude)
        plot_list = [image, magnitude, power_spectrum]
        subs = ['Image', 'Magnitude', 'Power Spectrum (zero shifted)']
        imshow_all(plot_list, subtitles=subs, out_fname=f"power_sectrum_{i}.png")

log_1p = lambda x : np.log1p(np.abs(x))

def do_gaussian_filtering(images, radius=30):
    for i, image in enumerate(images):
        H = gaussian_lp_filter(image, d_0=radius)
        F = np.fft.fft2(image)
        F_shifted = np.fft.fftshift(F)
        G_shifted = F_shifted * H
        G = np.fft.ifftshift(G_shifted)
        g = np.abs(np.fft.ifft2(G))
        imshow_all([image, g], subtitles=['Original Image', 'Gaussian LP filtered image'], out_fname=f'Gauss_fil_{i}.png')

def do_ideal_lowpass_filtering(images, radius=20):
    for i, image in enumerate(images):
        H = ideal_lp_filter(image, d_0=radius)
        F = np.fft.fft2(image)
        F_shifted = np.fft.fftshift(F)
        G_shifted = F_shifted * H
        G = np.fft.ifftshift(G_shifted)
        g = np.abs(np.fft.ifft2(G))
        imshow_all([image, g], subtitles=['Original Image', 'Ideal LP filtered image'], out_fname=f'Ideal_lp_fil_{i}.png')


# < ----------- Part 2 funcs -------------- >


def get_max_phase_correlation(f, g):
    F, G = np.fft.fft2(f), np.fft.fft2(g)
    F_conj = np.conj(F)
    matrix = F_conj * G / (np.abs(F_conj) * np.abs(G))
    phase_corr = np.fft.ifft2(matrix)
    phase_corr_abs = np.abs(phase_corr)
    maxx = phase_corr_abs.max()
    return maxx

def get_phase_correlation_peak(f, g, d_0=50):
    H = gaussian_lp_filter(g, d_0=d_0)
    F, G = np.fft.fft2(f), np.fft.fft2(g)
    G_shifted = np.fft.fftshift(G)
    G_lp = H * G_shifted
    G_filtered = np.fft.ifftshift(G_lp)
    F_conj = np.conj(F)
    matrix = F_conj * G_filtered / (np.abs(F_conj) * np.abs(G))
    phase_corr = np.fft.ifft2(matrix)
    phase_corr_abs = np.abs(phase_corr)
    maxx = phase_corr_abs.max()
    peak = np.argwhere(phase_corr_abs == maxx)
    return peak[0], maxx

def make_phase_corr_matrix(images):
    n = len(images)
    pcm = [[None for r in range(n)] for c in range(n)]
    for r in range(n):
        for c in range(n):
            if r == c:
                pcm[r][c] = ((-1, -1), 0)
                continue
            ref_img, other_img = images[r], images[c]
            peak, max_crr = get_phase_correlation_peak(other_img, ref_img)
            pcm[r][c] = (list(peak), max_crr)

    return pcm


def get_max_corr_img(num, pcm, placed_list):
    """
        Returns the image number and its peak which is best correlated with the image trying to place. 
    """
    maxi = float("-inf")
    its_peak = None
    img_idx = None
    for idx in range(len(pcm[num])):
        if idx == num:
            continue
        if idx in placed_list:
            continue
        peak, max_corr = pcm[num][idx]
        if max_corr > maxi:
            maxi = max_corr
            its_peak = peak
            img_idx = idx

    return its_peak, img_idx    
    

def get_max_corr_dict(pcm):
    max_corrs = []
    for row in range(len(pcm)):
        cnt = 0
        for col in range(len(pcm[0])):
            if row == col:
                continue
            _, max_corr = pcm[row][col]
            if max_corr > CORR_THR:
                cnt += 1
        max_corrs.append((row, cnt))
    return max_corrs

def getMSE(a,b):
    return np.mean((a-b)**2)


def get_overlap_position(other_img, ref_img, peak):
    """
        Gets the position where the other image should be place w.r.t reference image.
        returns:  
            1 if other image to be place on the peak of the phase correlation  
            2 if other image at an offset Y from the peak
            3 if other image at an offset X from the peak
            4 if other image at an offse -X -Y from the peak
    """
    region = np.hstack((ref_img, ref_img))
    region = np.vstack((region, region))
    best = float("-inf")
    position = None
    dx, dy = peak
    X, Y = other_img.shape
    ref_region_1 = region[dx: X,dy: Y]
    other_img_1 = other_img[ 0: X-dx, 0: Y-dy]
    corr1 = get_max_phase_correlation(other_img_1, ref_region_1)
    if corr1 > best:
        position = 1
        best = corr1

    ref_region_2 = region[dx: X, Y:Y+dy]
    other_img_2 = other_img[0:X-dx,Y-dy:]
    corr2 = get_max_phase_correlation(other_img_2, ref_region_2)
    if corr2 > best:
        position = 2
        best = corr2

    ref_region_3 = region[X:X+dx, dy: Y]
    other_img_3 = other_img[X-dx:, 0:Y-dy]
    corr3 = get_max_phase_correlation(other_img_3, ref_region_3)
    if corr3 > best:
        position = 3
        best = corr3

    ref_region_4 = region[X:X+dx, Y:Y+dy]
    other_img_4 = other_img[X-dx:, Y-dy:]
    corr4 = get_max_phase_correlation(other_img_4, ref_region_4)
    if corr4 > best:
        position = 4
        best = corr4

    return position


def place_image(canvas, img, top_left):
    """
        Place image on the canvas
    """
    start_x, start_y = top_left
    X, Y = img.shape
    canvas[start_x: start_x + X, start_y: start_y + Y] = img


def build_mosaic(cell_imgs):
    """
        Builds an empty white canvas and places each image in its place
    """
    pcm = make_phase_corr_matrix(cell_imgs)
    # corrs = get_max_corr_dict(pcm)
    H, W = cell_imgs[0].shape
    canvas = np.ones((H*5, W*5)) * 255
    first_img_num = 2
    first_img = cell_imgs[first_img_num]
    canvas_mid = canvas.shape[0]//2, canvas.shape[1]//2
    start = canvas_mid[0] - H//2, canvas_mid[1] - W//2
    place_image(canvas, first_img, start)
    placed_list = [first_img_num]


    while len(placed_list) < len(cell_imgs):
        ref_img_num = placed_list[-1]
        # pdb.set_trace()
        ref_img = cell_imgs[ref_img_num]
        peak, img_to_place_idx = get_max_corr_img(ref_img_num, pcm, placed_list)
        img_to_place = cell_imgs[img_to_place_idx]
        which_position = get_overlap_position(img_to_place, ref_img, peak)

        peak_offset_x, peak_offset_y = peak
        the_start_x, the_start_y = start
        offset_x, offset_y = POSITION_TO_OFFSET_MAP[which_position]
        new_start = (the_start_x + peak_offset_x + offset_x * H, the_start_y + peak_offset_y + offset_y * W)
        place_image(canvas, img_to_place, new_start)
        placed_list.append(img_to_place_idx)
        start = new_start

    return canvas

    

def get_cell_images():
    cell_imgs = []
    current_working_directory = os.getcwd()
    image_path  = current_working_directory + '/cell_images/'
    with open('./cell_images/read.txt') as f:
        for line in f.readlines():
            f_name = line.split()[0]
            cell_imgs.append(imread(image_path + f_name, as_gray=True))
    return cell_imgs

if __name__ == "__main__":
    # Create output dir if not exists
    current_working_directory = os.getcwd()
    output_dir = current_working_directory + '/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Part 1 FFT 

    fft_images = get_fft_images()

    
    plot_image_power_spectrum(fft_images)

    # plt.show()

    # Part 1 Filtering in frequency domain

    do_gaussian_filtering(fft_images)

    # plt.show()

    do_ideal_lowpass_filtering(fft_images)

    # plt.show()

    # Part 2 Mosaic building

    cell_images = get_cell_images()

    print("Building image mosaic, please wait .....")

    mosaic = build_mosaic(cell_images)
    imshow_img(mosaic, out_fname='mosaic_cells.png')
    plt.show()

import sys
import os
import cv2
from skimage import img_as_float, filters
from skimage.restoration import denoise_nl_means
import numpy as np
import tensorflow as tf
from PIL import Image, PngImagePlugin


def non_local_means(img):
    img_float = img_as_float(img)  # converts to float in [0,1]

    h = 7
    templ = 3
    search = 3

    return denoise_nl_means(
        img_float,
        h=h / 255.0,  # normalize h to [0,1] scale
        patch_size=templ,
        patch_distance=search // 2,
        fast_mode=True,
        channel_axis=None
    )


def unsharp_masking(img):
    amount = 0.6

    return filters.unsharp_mask(img, radius=5, amount=amount)


def make_gaussian_psf(shape_img, sigma=(2.5, 2.5), size=(15, 15), norm='peak'):
    H, W = shape_img
    sy, sx = sigma
    ky, kx = size
    y = np.arange(ky) - (ky - 1) / 2.0
    x = np.arange(kx) - (kx - 1) / 2.0
    Y, X = np.meshgrid(y, x, indexing='ij')
    psf_small = np.exp(-(Y**2 / (2 * sy**2 + 1e-12) + X**2 / (2 * sx**2 + 1e-12)))
    if norm == 'sum':
        psf_small /= psf_small.sum()
    else:  # 'peak'
        psf_small /= psf_small.max()

    psf_full = np.zeros((H, W), dtype=np.float32)
    y0 = (H - ky) // 2
    x0 = (W - kx) // 2
    psf_full[y0:y0+ky, x0:x0+kx] = psf_small
    psf_full = np.fft.ifftshift(psf_full).astype(np.float32)

    return psf_full


def fft_convolve(x, otf):
    X = tf.signal.fft2d(tf.cast(x, tf.complex64))
    Y = X * otf
    y = tf.signal.ifft2d(Y)
    return tf.math.real(y)


def edgetaper2d(img, alpha=0.2):
    """Simple Tukey-like apodization to reduce FFT wrap artifacts."""
    h, w = img.shape
    y = np.hanning(max(2, int(alpha*h)))
    x = np.hanning(max(2, int(alpha*w)))
    wy = np.ones(h); wy[:len(y)//2] = y[:len(y)//2]; wy[-len(y)//2:] = y[-len(y)//2:]
    wx = np.ones(w); wx[:len(x)//2] = x[:len(x)//2]; wx[-len(x)//2:] = x[-len(x)//2:]
    window = np.outer(wy, wx)
    window = window / window.mean()
    return img * window


def richardson_lucy_deconvolution(img, output_dtype="uint16", edgetaper=True, eps=1e-7):
    iterations = 2

    h, w = img.shape
    border = max(int(0.2*h), int(0.2*w))

    img_uint16 = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_REPLICATE)

    # Normalize input to [0,1]
    y = img_uint16.astype(np.float32) / 65535.0
    if edgetaper:
        y = edgetaper2d(y, alpha=0.2)

    # Build PSF and OTF
    psf = make_gaussian_psf(y.shape)
    psf_tf = tf.convert_to_tensor(psf)
    otf = tf.signal.fft2d(tf.cast(psf_tf, tf.complex64))
    otf_conj = tf.math.conj(otf)

    # RL iterations
    x = tf.convert_to_tensor(y, dtype=tf.float32)
    for _ in range(iterations):
        conv = fft_convolve(x, otf)
        ratio = y / (conv + eps)
        corr = fft_convolve(tf.convert_to_tensor(ratio), otf_conj)
        x = x * corr
        x = tf.nn.relu(x)

    x = x.numpy()
    imin, imax = float(x.min()), float(x.max())

    if imax > imin:
        if output_dtype == "uint16":
            out = ((x - imin) / (imax - imin) * 65535.0 + 0.5).astype(np.uint16)
        else:  # fallback to uint8
            out = ((x - imin) / (imax - imin) * 255.0 + 0.5).astype(np.uint8)
    else:
        out = np.zeros_like(x, dtype=np.uint16 if output_dtype == "uint16" else np.uint8)

    out = out[border:-(border), border:-(border)]
    return out


def histogram_stretch(img, sat_percent=0.5):
    if img.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image.")

    # Determine range based on dtype
    print(img.dtype)
    if img.dtype == np.uint8:
        out_dtype = np.uint8
        max_val = 255.0
    elif img.dtype == np.uint16:
        out_dtype = np.uint16
        max_val = 65535.0
    else:
        raise ValueError("Unsupported image dtype. Must be uint8 or uint16.")

    img_float = img.astype(np.float32)

    # Find low and high cut-off values using percentiles
    low_val = np.percentile(img_float, sat_percent)
    high_val = np.percentile(img_float, 100 - sat_percent)

    # Avoid division by zero if image is flat
    if high_val == low_val:
        return np.zeros_like(img, dtype=out_dtype)

    # Stretch to full [0, max_val] range
    stretched = np.clip((img_float - low_val) * max_val / (high_val - low_val), 0, max_val)

    return stretched.astype(out_dtype)


def tescan_digital_resolution_filter(img):
    img = non_local_means(img)
    img = unsharp_masking(img)
    img = richardson_lucy_deconvolution(img)
    img = histogram_stretch(img)
    return img


def get_all_files(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.hdr') and not f.endswith('.db') and not f.endswith('.txt')]
    res_paths = []

    for file in files:
        whole_path = os.path.join(path, file)
        if os.path.isdir(whole_path):
            res_paths += get_all_files(whole_path)
        else:
            res_paths.append(whole_path)

    return res_paths


# Runs TESCAN Digital Resolution on image on the input path. If input path is a folder, the filter is applied on all images in the folder.
# Result is stored next to the input image with the sufix '_enhanced'
def main():
    # Check if user provided a path
    if len(sys.argv) < 2:
        print("Usage: python tescan_digital_resolution.py \"<path>\"")
        return

    path = sys.argv[1]

    print(f"You entered the path: {path}")

    if os.path.isdir(path):
        files = get_all_files(path)
    else:
        files = [path]

    for p in files:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

        img = tescan_digital_resolution_filter(img)

        new_name = '.'.join(p.split('.')[:-1]) + f'_enhanced.png'
        cv2.imwrite(new_name, img)


if __name__ == '__main__':
    main()

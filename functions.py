#!/usr/bin/env python

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#%% subfunctions
def rgb_to_gray_lightness(imgRgb): # Lightness method
    return (np.min(imgRgb[..., :3], axis=-1) + np.max(imgRgb[..., :3], axis=-1)) * 0.5

def rgb_to_gray_average(imgRgb): # Average method
    return np.dot(imgRgb[..., :3], [0.3333, 0.3333, 0.3333])

def rgb_to_gray_luminosity(imgRgb): # Luminosity method(general method), same result with tensorflow.image.rgb_to_grayscale()
    return np.dot(imgRgb[..., :3], [0.2989, 0.5870, 0.1140])

def load_image_as_rgb_and_gray8bits(fileName, method="luminosity"):
    max8bits = 2**8 - 1
    max16bits = 2**16 - 1
    try:
        imgRgb = plt.imread(fileName)
    except Exception:
        raise Exception("Unsupported image format of the file '{}'.".format(fileName))
    if imgRgb.dtype == np.uint8:
        imgRgb = imgRgb.astype(np.float32) / max8bits
    elif imgRgb.dtype == np.uint16:
        imgRgb = imgRgb.astype(np.float32) / max16bits
    elif imgRgb.dtype == np.float16:
        imgRgb = imgRgb.astype(np.float32)
    elif imgRgb.dtype == np.float32 or imgRgb.dtype == np.float64:
        pass
    else:
        raise ValueError("Unsupported image data type")
    if method == "lightness":
        return imgRgb, np.around(rgb_to_gray_lightness(imgRgb) * max8bits).astype("uint8")
    elif method == "average":
        return imgRgb, np.around(rgb_to_gray_average(imgRgb) * max8bits).astype("uint8")
    elif method == "luminosity":
        return imgRgb, np.around(rgb_to_gray_luminosity(imgRgb) * max8bits).astype("uint8")
    else:
        raise ValueError("Unsupported rgb2gray method")

def get_brightness_profile_as_list(imageGray, targetColumn=0, numBptsInt=16, isShow=False, figTitle=""):
    profile = imageGray[targetColumn, :].tolist()
    srcLeng = len(profile)
    if srcLeng < numBptsInt:
        raise ValueError("A number of breakpoints is too many compared to a list size")
    bptsList = np.around(np.linspace(0, srcLeng - 1, numBptsInt + 1, endpoint=True)[:numBptsInt] + (srcLeng / numBptsInt * 0.5)).astype("int32").tolist()
    if isShow:
        if targetColumn == 0:
            textSpecDict = {"ha": 'left', "va": 'top', "fontsize": 8}
        else:
            textSpecDict = {"ha": 'left', "va": 'bottom', "fontsize": 8}
        plt.figure(figsize=(10, 5))
        plt.imshow(imageGray, cmap="gray", vmin=0, vmax=255)
        if targetColumn == -1:
            textYLoc = imageGray.shape[0] - 5
        else:
            textYLoc = targetColumn + 25
        for bpt in bptsList:
            plt.text(int(bpt - (srcLeng / numBptsInt * 0.5)), textYLoc, str(imageGray[targetColumn, bpt]), color='red')
        plt.title(figTitle)
    return bptsList, np.array(profile)[bptsList]

def regression(x, y, X, method, skipClpSat=False):
    """
    This function performs interpolation or regression on the given x-y data points, and returns the dependent variables on
    the given new x points using the specified method.
    Parameters:
    x: list of x values
    y: list of y values
    X: list of x values to calculate the interpolated or regressed y values
    method: str, interpolation or regression method to be used ('linear', 'cubic', or 'spline')
    skipClpSat: bool, optional, whether to remove saturated (255) or clipped out (0) data points from dependent variables of these regressions
    Returns:
    x: list of x values
    y: list of y values
    X: list of x values for which y values were interpolated or regressed
    YValid: list of valid (not over/undershot, not decreased) y values
    H: regression or interpolation object
    errIdx: list of x values for which y values are invalid (i.e., outside the range of previous values)
    """
    min8bits = 0
    max8bits = 2**8 - 1
    if not(len(x) == len(y)):
        raise ValueError("length is not matched for x and y")
    if skipClpSat:
        idx = (y > min8bits) & (y < max8bits)
        x = np.array(x)[idx].tolist()
        y = np.array(y)[idx].tolist()
    if method == "linear":
        H = np.polyfit(x, y, deg=1)
        Y = np.polyval(H, X)
    elif method == "cubic":
        H = interpolate.interp1d(x, y, kind="cubic", fill_value="extrapolate")
        Y = H(X)
    elif method == "spline":
        H = interpolate.UnivariateSpline(x, y)
        Y = H(X)
    else:
        raise ValueError("Unsupported interpolation method")
    Y = np.clip(np.around(Y), min8bits, max8bits).astype("uint8").tolist()
    YValid = list()
    errIdx = list()
    for valx, valy in zip(X, Y):
        if not(YValid):
            YValid.append(valy)
        else:
            if valy >= YValid[-1]:
                YValid.append(valy)
            else:
                YValid.append(YValid[-1])
                errIdx.append(valx)
    return x, y, X, YValid, H, errIdx

def piecewiseLinearInterpolation(x, y):
    min8bits = 0
    max8bits = 2**8 - 1
    if not(len(x) == len(y)):
        raise ValueError("length is not matched for x and y")
    X = np.linspace(min(x), max(x), max(x) - min(x) + 1, endpoint=True).astype("uint8").tolist()
    H = interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")
    Y = H(X)
    return x, y, X, np.clip(np.around(Y), min8bits, max8bits).astype("uint8").tolist()

def solve_inverse_problem(x, h):
    min8bits = 0
    max8bits = 2**8 - 1
    xi = np.linspace(min8bits, max8bits, max8bits - min8bits + 1, endpoint=True).astype("uint8").tolist()
    yi = list()
    for i in xi:
        yi.append(solve_for_x(x, i, h))
    return xi, np.around(yi).astype("uint8").tolist()

def solve_for_x(x, y, h): # bisection method
    from scipy.optimize import root_scalar
    x_start = x[0]
    x_end = x[-1]
    try:
        if callable(h):
            while h(x_start) >= y:
                x_start += 1
            while h(x_end) <= y:
                x_end -= 1
            return root_scalar(lambda x: h(x) - y, bracket=[x_start, x_end]).root
        else:
            while np.polyval(h, x_start) >= y:
                x_start  += 1
            while np.polyval(h, x_end) <= y:
                x_end -= 1
            return root_scalar(lambda x: np.polyval(h, x) - y, bracket=[x_start, x_end]).root
    except:
        return 0

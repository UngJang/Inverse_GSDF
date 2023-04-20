#!/usr/bin/env python

#%% imports
import json
from functions import *

GsdfImagePathName = "./Source/versana-gray crop.png"
TargetNumBpts = 16
rgb2grayMethod = "luminosity" # "luminosity", "average", "luminosity"
regressMethod = "cubic" # "linear", "cubic", "spline"
regressOnlyValid = True # True means "Patial", False means "Full"
ExtFileName = "./Result/LUT_Inverse_GSDF_partialCubic.json"
isDebug = True
saveLut = True


if __name__ == "__main__":
    # load an image as a 8-bits grayscale
    imageRgb, imageGray8Bits = load_image_as_rgb_and_gray8bits(GsdfImagePathName, method=rgb2grayMethod)
    # generate the reference brightness profile (assumed, guided by SJLee)
    refBrightnessList = np.linspace(0.0, 2**8 - 1.0, TargetNumBpts, endpoint=True).astype("uint8").tolist()
    # get brightness profiles from the 8-bits grayscale image
    _, gsdfBrightnessList = get_brightness_profile_as_list(imageGray8Bits, targetColumn=0, numBptsInt=TargetNumBpts)
    # do cubic regression
    GSDF_X = np.arange(2**8).astype("uint8").tolist()
    _, _, _, GSDF_Y, GSDF_H, _ = regression(refBrightnessList, gsdfBrightnessList, GSDF_X, method=regressMethod, skipClpSat=regressOnlyValid)
    # get inverse GSDF curve
    invGSDF_X, invGSDF_Y = solve_inverse_problem(GSDF_X, GSDF_H)
    # draw images
    _, _ = get_brightness_profile_as_list(imageGray8Bits, targetColumn=0, numBptsInt=TargetNumBpts, isShow=True, figTitle="GSDF image (upper row)")
    _, _ = get_brightness_profile_as_list(np.array(invGSDF_Y)[imageGray8Bits], targetColumn=0, numBptsInt=TargetNumBpts, isShow=True, figTitle="Inversed GSDF image (upper row)")
    if isDebug:
        plt.figure()
        plt.plot(GSDF_Y, GSDF_X, label="swapped drawing between x and y (ref)")
        plt.plot(invGSDF_X, invGSDF_Y, color="red", linewidth=3, linestyle='--', label="estimated inverse gsdf curve")
        plt.legend()
        plt.grid()
    plt.show()
    if saveLut:
        inverse_GSDF_Lut = {"rgb2grayMethod": rgb2grayMethod,
                            "regressMethod": regressMethod,
                            "regressOnlyValid": regressOnlyValid,
                            "x": invGSDF_X,
                            "y": invGSDF_Y,} 
        with open(ExtFileName, 'w', encoding="UTF8") as fileToWrite:
            fileToWrite.write(json.dumps(inverse_GSDF_Lut, indent=4))
            fileToWrite.close()

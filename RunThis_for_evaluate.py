#!/usr/bin/env python

#%% imports
from functions import *

GsdfImagePathName = "./Source/versana-gray crop.png"
TargetNumBpts = 16

if __name__ == "__main__":
    # load an image as a 8-bits grayscale
    imageRgb, imageGray8Bits = load_image_as_rgb_and_gray8bits(GsdfImagePathName, method="luminosity")
    # generate the reference brightness profile (assumed, guided by SJLee)
    refBrightnessList = np.linspace(0.0, 2**8 - 1.0, TargetNumBpts, endpoint=True).astype("uint8").tolist()
    # get brightness profiles from the 8-bits grayscale image
    _, topProfileBrightnessList = get_brightness_profile_as_list(imageGray8Bits, targetColumn=0, numBptsInt=TargetNumBpts)
    _, btmProfileBrightnessList = get_brightness_profile_as_list(imageGray8Bits, targetColumn=-1, numBptsInt=TargetNumBpts)
    btmProfileBrightnessList = btmProfileBrightnessList[::-1] # it looks a bottom profile has inversed indice
    gsdfBrightnessList = topProfileBrightnessList
    #%% various regressions (linear regression: linear, skipped linear; non-linear regressions: cubic, spline, skipped cubic, skipped spline)
    # do regressions
    xList = np.arange(2**8).astype("uint8").tolist()
    _, _, _, yFullLinear, hFullLinear, errIdxFullLinear = regression(refBrightnessList, gsdfBrightnessList, xList, method="linear", skipClpSat=False)
    _, _, _, yFullCubic, hFullCubic, errIdxFullCubic = regression(refBrightnessList, gsdfBrightnessList, xList, method="cubic", skipClpSat=False)
    _, _, _, yFullSpline, hFullSpline, errIdxFullSpline = regression(refBrightnessList, gsdfBrightnessList, xList, method="spline", skipClpSat=False)
    _, _, _, yPartialLinear, hPartialLinear, errIdxPartialLinear = regression(refBrightnessList, gsdfBrightnessList, xList, method="linear", skipClpSat=True)
    xEval_bpts, yEval_bpts, _, yPartialCubic, hPartialCubic, errIdxPartialCubic = regression(refBrightnessList, gsdfBrightnessList, xList, method="cubic", skipClpSat=True)
    _, _, _, yPartialSpline, hPartialSpline, errIdxPartialSpline = regression(refBrightnessList, gsdfBrightnessList, xList, method="spline", skipClpSat=True)
    _, _, xEval_pwlinear, yEval_pwlinear = piecewiseLinearInterpolation(xEval_bpts, yEval_bpts)
    # calc a difference
    yList = [yFullLinear, yFullCubic, yFullSpline, yPartialLinear, yPartialCubic, yPartialSpline]
    difList_bpts = [np.array(f)[xEval_bpts] - np.array(yEval_bpts) for f in yList]
    difList_pwlinear = [np.array(f)[xEval_pwlinear] - np.array(yEval_pwlinear) for f in yList]
    # calc an error (std)
    stdList_bpts = [np.std(dif) for dif in difList_bpts]
    stdList_pwlinear = [np.std(dif) for dif in difList_pwlinear]
    # calc the RMSE
    rmsList_bpts = [np.sqrt(np.mean(dif**2)) for dif in difList_bpts]
    rmsList_pwlinear = [np.sqrt(np.mean(dif**2)) for dif in difList_pwlinear]
    #%% draw
    # estimated GSDF curves
    mainPltSpecDict = {"num": 98, "nrows": 2, "ncols": 3, "figsize": (18, 9)}
    subplotSpecDict = {"left": 0.05, "bottom": 0.05, "right": 0.95, "top": 0.95, "wspace": 0.2, "hspace": 0.3}
    makersSpectDict = {"marker": 'x', "s": 100, "c": 'green', "label": "Decrement happened"}
    refDrawSpecList = [refBrightnessList, gsdfBrightnessList, '-ok']
    refDrawSpecDict = {"label": 'True', "linewidth": 2}
    tarDrawSpecDict = {"linewidth": 1}
    fig, axes = plt.subplots(**mainPltSpecDict)
    fig.subplots_adjust(**subplotSpecDict)
    titleList = ['Linear', 'Cubic', 'Spline', 'Partial linear', 'Partial cubic', 'Partial spline']
    errIdxListList = [errIdxFullLinear, errIdxFullCubic, errIdxFullSpline, errIdxPartialLinear, errIdxPartialCubic, errIdxPartialSpline]
    for idx, tmpAxis in enumerate(axes.flat):
        tmpAxis.plot(*refDrawSpecList, **refDrawSpecDict)
        tmpAxis.plot(xList, yList[idx], '-r', label=titleList[idx], **tarDrawSpecDict)
        tmpAxis.grid()
        tmpAxis.set_title(titleList[idx])
        if errIdxListList[idx]:
            tmpAxis.scatter(errIdxListList[idx], np.array(yList[idx])[errIdxListList[idx]], **makersSpectDict)
        tmpAxis.legend()
        tmpAxis.set_aspect('equal')
    plt.tight_layout()
    # RMSE
    mainPltSpecDict = {"num": 99, "nrows": 1, "ncols": 2, "figsize": (12, 6)}
    subplotSpecDict = {"left": 0.05, "bottom": 0.05, "right": 0.95, "top": 0.95, "wspace": 0.2, "hspace": 0.3}
    errbarsSpecDict = {"fmt": "o", "capsize": 3}
    errTextSpecDict = {"ha": 'right', "va": 'bottom', "fontsize": 8}
    makersSpectDict = {"marker": 'o', "s": 100, "linewidth": 2, "edgecolors": 'red', "facecolors": "None", "label": "Minimum RMSE"}
    fig, axes = plt.subplots(**mainPltSpecDict)
    fig.subplots_adjust(**subplotSpecDict)
    rmsListList = [rmsList_bpts, rmsList_pwlinear]
    stdListList = [stdList_bpts, stdList_pwlinear]
    titleList = ["Error calculated at breakpoints only", "Error of deviation from piecewise linear model"]
    xlabelList = ["Linear", "Cubic", "Spline", "Partial\nLinear", "Partial\nCubic", "Partial\nSpline"]
    for idx0, tmpAxis in enumerate(axes.flat):
        tmpAxis.errorbar(xlabelList, rmsListList[idx0], yerr=stdListList[idx0], **errbarsSpecDict)
        tmpAxis.grid()
        tmpAxis.set_xlabel("Regression model")
        tmpAxis.set_ylabel("RMSE")
        tmpAxis.set_title(titleList[idx0])
        tmpAxis.set_ylim(-0.5, max(rmsListList[idx0]) + max(stdListList[idx0]) + 1.0)
        for idx1, (tmpXLabel, tmpRms, tmpStd) in enumerate(zip(xlabelList, rmsListList[idx0], stdListList[idx0])):
            tmpText = tmpAxis.text(tmpXLabel, tmpRms + tmpStd + 0.1, f"RMSE = {tmpRms:.3f}\nSTD = {tmpStd:.3f}", **errTextSpecDict)
            if idx1 == 0:
                tmpText.set_ha("left")
            if tmpRms == min(rmsListList[idx0]):
                tmpAxis.scatter(tmpXLabel, tmpRms, **makersSpectDict)
    tmpAxis.legend()
    plt.tight_layout()
    plt.show()

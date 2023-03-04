#!/usr/bin/env python
# coding: utf-8

# Notebook for housing and finalizing the spark detection routines
# 
# - other notebook was chowing down on memory so I decided to switch to doing the analysis one at a time

import matplotlib.pyplot as plt
import tifffile
import numpy as np
from scipy import ndimage
import ipywidgets as widgets
from skimage import measure
#import cv2
import copy
import scipy

plt.rcParams['figure.figsize'] = [12,12]


def convToUm(dist_in_pxs):
  '''Function to convert a distance given in pixels to a distance in microns based on scope resolution
  Note: Resolution taken from 19.lsm but assumed to be the same'''
  resolution = 6.2596 # pixels per micron
  dist_in_ums = dist_in_pxs * (1./resolution)
  return dist_in_ums

### define gaussian fitting function for fitting spark width at half maximum
def twoD_Gaussian(A, amplitude, sigma, theta, offset):
  x, y, xo, yo = A
  xo = float(xo)
  yo = float(yo)    
  a = (np.cos(theta)**2)/(2*sigma**2) + (np.sin(theta)**2)/(2*sigma**2)
  b = -(np.sin(2*theta))/(4*sigma**2) + (np.sin(2*theta))/(4*sigma**2)
  c = (np.sin(theta)**2)/(2*sigma**2) + (np.cos(theta)**2)/(2*sigma**2)
  g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
  return g.ravel()

# ### 0. Specify Arguments and Preprocess Data
# 
# For Preprocessing:
# 
# - clip time series data such that there are no significant contractions and that there aren't any calcium waves. This throws off the preprocessing routine where we average to denoise
#     - For this data I clipped:
#         - 3.lsm at frame 1810
#         - 19.lsm at frame 2862

clipImages = True
plotMe = False
thresholdValue = 3.
root = "/net/share/dfco222/data/Xins_CaSparkData/NewData/"
#fileName = "10.lsm"
#fileName = "6.lsm"
#fileName = "19.lsm"

def detectAndAnalyzeSparks_quick(fileName, 
                                 thresholdValue=3.,
                                 plotMe=False,
                                 verbose=False,
                                 readIntermediateFiles=True,
                                 writeIntermediateFiles=True,
                                 ):

    ## Save the path to the files for later saving
    pathToFile = fileName.split('/')

    if not readIntermediateFiles:
        # ### 1. Read in Image and Clip as Needed
        #     

        ### Tifffile has some weird way of reading in image in much higher dimension array than we need so we just select [0,0,:,:,:]
        if fileName[-4:] == '.lsm':
            img = tifffile.imread(fileName)[0,0,:,:,:]
        else:
            img = tifffile.imread(fileName)
            #print(np.shape(img))

        imgDims = np.shape(img)

        # ### 2. Smooth Image by Averaging Stack in Time-Dimension
        #largeSmoothed = np.mean(img,axis=0)
        ### Reshape to be the same size as the image (Essentially extruding in the time dimension)
        #largeSmoothed = np.broadcast_to(largeSmoothed,imgDims)

        # ### 2. Smooth Image by Using Large Filter 
        ### The 120 size of the filter is ad hoc and could be subject to change
        ## Sparks are 20-40 ms duration half max so 120 should be large enough to not affect those transients
        import time
        start = time.time()
        largeKernel = np.ones((120,3,3),dtype=np.float)
        largeKernel /= np.sum(largeKernel)
        largeSmoothed = ndimage.convolve(img,largeKernel,mode='mirror')
        print("Total time to run large smoothing:", time.time()-start)
        # ### 3. Smooth Images using a Small Kernel to Denoise the Calcium Transients Without Smoothing them Out

        kernel = np.ones((3,5,5),dtype=np.float32)
        kernel /= np.sum(kernel)

        smallSmoothed = ndimage.convolve(img,kernel,mode='mirror')
    else:
        largeName = '/'.join(pathToFile[:-1])+'/largeSmoothed/'+fileName
        smallName = '/'.join(pathToFile[:-1])+'/smallSmoothed/'+fileName

        largeSmoothed = tifffile.imread(largeName)
        smallSmoothed = tifffile.imread(smallName)


    # ### 4. Characterize Noise and Obtain Segmented Image of Cardiomyocyte

    noiseImg = img.copy() - largeSmoothed.copy()
    ### Getting the standard deviation of the noise
    noiseStd = np.std(noiseImg)

    ### Segmenting cardiomyocyte from the background based on noise characteristics
    segmented = largeSmoothed > (3 * noiseStd + np.mean(noiseImg))

    ### Perform morphological opening to deal with any noise specks that may have found their way into the data
    segmented = ndimage.morphology.binary_opening(segmented,
                                                  structure=np.ones((1,7,7)))

    if plotMe:
        plt.figure()
        plt.imshow(segmented[100,:,:])
        plt.show()
    ### Perform some dilation to ensure we grab all of the myocyte?
    ## This didn't seem to be necessary previously

    if verbose:
        print("Noise Standard Deviation =",noiseStd)

    # ### 5. Divide Out the Long Time Average Image from the Short Time Average Image to Denoise the image

    transientImg = smallSmoothed.astype(np.float64) / largeSmoothed.astype(np.float64)

    ### Apply mask from segmented image
    transientImg[segmented == 0] = 0


    # ### 6. Use Sobel Filter in the Time Direction to Pull Out Portions of the Image that Have High Gradients (Sparks)
    # Sparks seem to appear within a frame or two and are typically large intensity jumps. Should be able to pull this out really well with a 3D Sobel Filter
    timeGrad = ndimage.sobel(transientImg, axis=0, mode='nearest')
    #tifffile.imsave(fileName[:-4]+'_timeGradient.tif',data=timeGrad)

    if plotMe:
        def pltF(f):
            plt.imshow(timeGrad[f,:,:],cmap='gray')
            plt.colorbar()
            plt.show()
        widgets.interact(pltF, f=widgets.IntSlider(min=0,max=imgDims[0]-1,step=1,value=0,layout=widgets.Layout(width='100%')))

        #thresholded = timeGrad.copy()
        #thresholded[thresholded < thresholdValue] = 0
        #def pltF(f):
        #    plt.imshow(thresholded[f,:,:],cmap='gray')
        #    plt.colorbar()
        #    plt.show()
        #widgets.interact(pltF, f=widgets.IntSlider(min=0,max=imgDims[0]-1,step=1,value=0,layout=widgets.Layout(width='100%')))


    # ### 7. Threshold to Pull Out Sparks

    caSparks = timeGrad.copy() > thresholdValue


    # ### 8. Perform Opening to Disregard Small Detections That are Likely Noise

    caSparks = ndimage.morphology.binary_opening(caSparks,
                                                 structure=np.ones((2,2,2)))


    # ### 9. Dilate in Time Dimension so We Get Better Context of Spark

    caSparks = ndimage.morphology.binary_dilation(caSparks,
                                                  structure=np.ones((5,1,1)))


    # ### 10. Label and Delineate Segmented Sparks

    labeledSparks = measure.label(caSparks.copy(),background = 0)
    if verbose:
        print("Number of Detected Sparks:", np.max(labeledSparks))

    # ### 11. Pull Out Sparks, Average, and Plot Spark Characteristics

    ### prototyping for right now
    ## spark number we're concerned with for right now
    sparkNum = 0 

    ## Mask Transient Image with spark Detections
    maskedTransientImg = transientImg

    ## get region
    wholeRegion = measure.regionprops(labeledSparks)

    numSparkDetections = np.max(labeledSparks)
    if numSparkDetections > 10:
        numSparkDetections = 10

    sparks = {}
    for sparkNum in range(numSparkDetections):
        ### Get region for spark number
        thisRegion = wholeRegion[sparkNum]

        ### Get coordinates for bounding box
        boundingCoordinates = np.asarray(thisRegion.bbox,dtype=np.uint64)
        if verbose:
            print("Bounding Coordinates", boundingCoordinates)
        ### Convert to easily referenced format
        sparkWindow = [[boundingCoordinates[0],boundingCoordinates[3]],
                       [boundingCoordinates[1],boundingCoordinates[4]],
                       [boundingCoordinates[2],boundingCoordinates[5]]]
        
        ### Create holder array that contains NaN where no spark was detected and transient spark where it was detected
        sparkHolder = maskedTransientImg.copy()[sparkWindow[0][0]:sparkWindow[0][1],
                                                sparkWindow[1][0]:sparkWindow[1][1],
                                                sparkWindow[2][0]:sparkWindow[2][1]]

        sparkValue_1D_avg = np.nanmean(np.nanmean(sparkHolder,axis=1),axis=1)
        ### Clip spark array to cutoff after max is reached
        sparkValue_1D_avg = sparkValue_1D_avg[:np.argmax(sparkValue_1D_avg)]
        x = np.arange(len(sparkValue_1D_avg))

        if plotMe:
            plt.figure()
            plt.title('Spark Characteristics for Spark #{}'.format(sparkNum))
            plt.ylabel('Spark Intensity [AU]')
            plt.xlabel('Time [ms]')
            plt.plot(x,sparkValue_1D_avg)
            plt.show()

        ### Store in dictionary for analyis
        sparks[str(sparkNum)] = {'spark_array':sparkValue_1D_avg,
                                 'spark_coordinates':sparkWindow}

    # ### 12. Fit Exponential to Data to Find Time of Half Maximum and Time to Half Maximum

    for sparkNum, sparkDict in sparks.items():
        sparkArray = sparkDict['spark_array']
        x = np.arange(len(sparkArray))

        ### Consider clipping sparkArray to max value

        ### Fitting y = A * log(x) + B
        ### Which translates to y = Ae^{Bx}
        #coeffs = np.polyfit(x, np.log(sparkArray), 1, w=np.sqrt(sparkArray))
        import scipy
        fitFunction = lambda t,b,c: np.min(sparkArray) + b* np.exp(c*t)
        #fitFunction = lambda t,a,b,c: a + b* np.exp(c*t)
        coeffs =  scipy.optimize.curve_fit(fitFunction, x, sparkArray,maxfev = 3000)
        #print coeffs

        fitFunctionValues = fitFunction(x,coeffs[0][0],coeffs[0][1])

        if plotMe:
            plt.figure()
            plt.plot(x,sparkArray,label='Experimental')
            plt.plot(x,fitFunctionValues,label='Fit')
            plt.legend()
            plt.show()

        halfMax = (np.max(fitFunctionValues) + np.min(fitFunctionValues)) / 2.
        timeHalfMax = np.argmin( np.sqrt((sparkArray - halfMax)**2 )) - np.argmin(sparkArray)
        sparkAmplitude = np.max(sparkArray) / np.min(sparkArray)
        timeToPeak = len(sparkArray) - np.argmin(sparkArray)
        if verbose:
            print("Half Maximum Value:", halfMax)
            print("Time to Half Maximum:", timeHalfMax)
            print("Spark Amplitude:",sparkAmplitude)
            print("Time to Peak:",timeToPeak)

        sparkDict['half_maximum'] = halfMax
        sparkDict['time_to_half_maximum'] = timeHalfMax
        sparkDict['spark_amplitude'] = sparkAmplitude
        sparkDict['time_to_peak'] = timeToPeak


    # ### 13. Find Centroid of Spark so We Can Analyze How Close it is to Sarcolemmal Surfaces
    
    ### We need to generate a holder that has the distance calculated for each point in the original segmented image
    ### NOTE: This is assuming the myocyte contraction is minor
    seg = segmented[0,:,:].copy()
    seg = seg.astype(np.uint8)
    
    distanceArray = ndimage.morphology.distance_transform_edt(seg)
    
    if plotMe:
        plt.figure()
        plt.imshow(distanceArray)
        plt.show()
        
    for sparkNum, sparkDict in sparks.items():
        thisRegion = wholeRegion[int(sparkNum)]
        cx, cy = thisRegion.centroid[1:]
        cx, cy = int(round(cx)), int(round(cy))
        centroid = cx, cy
        
        dist = distanceArray[cx,cy]
        dist = convToUm(dist)
        if verbose:    
            print("Spark Number {} Centroid:".format(sparkNum),centroid)
            print("Distance to Sarcolemmal Surface:", dist)
        sparkDict['centroid'] = centroid
        sparkDict['dist_to_sarcolemma'] = dist
    
    
    # ### 14. Find the Time at Half Maximum
    # This could be really tricky. Almost like we'll need to apply a region growing method for the time gradient 

    for sparkNum, sparkDict in sparks.items():
        ### Pull out the final time we decided for the spark peak
        coords = sparkDict['spark_coordinates']
        finalSparkTime = coords[0][1]

        condition = True
        decayedSparkTime = finalSparkTime
        while condition:
            sparkValue = np.mean(transientImg[int(decayedSparkTime),
                                                coords[1][0]:coords[1][1],
                                                coords[2][0]:coords[2][1]])

            ### Check that region growing condition is still satisfied
            ## We want to continue growing the region as long as the current time gradient and this time gradient is negative
            # SUBJECT TO CHANGE
            condition = sparkValue > sparkDict['half_maximum']
            decayedSparkTime += 1

            ### Check if decayedSparkTime is still within image bounds
            if decayedSparkTime == imgDims[0]:
                decayedSparkTime -= 1
                break

            if verbose:
                print(decayedSparkTime)





        ### Show new calcium spark transient now
        thisFullSpark = transientImg.copy()[coords[0][0]:int(decayedSparkTime),
                                                coords[1][0]:coords[1][1],
                                                coords[2][0]:coords[2][1]]
        fullSpark1D = np.mean(np.mean(thisFullSpark,axis=1),axis=1)

        if plotMe:
            plt.figure()
            plt.plot(np.arange(len(fullSpark1D)),fullSpark1D)
            plt.show()

        sparkDict['decay_to_half_max'] = len(fullSpark1D)
        sparkDict['duration_half_max'] = sparkDict['decay_to_half_max'] - sparkDict['time_to_half_maximum']


    # ### 15. Analyze Full Width Half Maximum



    for sparkName, sparkDict in sparks.items():
        coords = np.asarray(sparkDict['spark_coordinates'])
        timeAtHalfMaximum = coords[0][0] + sparkDict['time_to_half_maximum']
        if verbose:
            print(coords)
        ## try fitting half maximum window to be a little bigger so the gaussian fit is more accurate
        #coords += 5
        wideningParameter = 5
        if coords[1][0] > wideningParameter:
            coords[1][0] -= wideningParameter
        if coords[1][1] + wideningParameter < imgDims[1]:
            coords[1][1] += wideningParameter
        if coords[2][0] > wideningParameter:
            coords[2][0] -=wideningParameter
        if coords[2][1] + wideningParameter < imgDims[2]:
            coords[2][1] += wideningParameter

        if verbose:
            print(coords)
        
        halfMaximumWindow = transientImg[int(timeAtHalfMaximum),
                                         int(coords[1][0]):int(coords[1][1]),
                                         int(coords[2][0]):int(coords[2][1])]
        shape = np.shape(halfMaximumWindow)
        x_0,y_0 = shape
        x = np.arange(x_0)
        y = np.arange(y_0)

        x,y = np.meshgrid(x,y)

        ### Fit gaussian to half maximum time
        maxPoint_row,maxPoint_col = np.unravel_index(np.argmax(halfMaximumWindow),shape)
        initial_guess = (np.max(halfMaximumWindow), 3, 0, 1)
        params, _ = scipy.optimize.curve_fit(twoD_Gaussian, (x,y,maxPoint_row,maxPoint_col), halfMaximumWindow.flatten(),p0=initial_guess)

        if verbose:
            print("Sigma:", params[1])

        #data_fitted = twoD_Gaussian((x,y,maxPoint_row,maxPoint_col), *params)

        if plotMe:
            fig,ax = plt.subplots(1,1)
            ax.imshow(halfMaximumWindow, origin='bottom')
            plt.show()


        sparkDict['sigma'] = params[1]


    # ### 16. Display All Parameters of Interest

    for sparkNum, sparkDict in sparks.items():
        print("Spark Number:", sparkNum)
        #for quality, quantity in sparkDict.items():
        #    print"\t"+quality+":",quantity
        print("\tTime to Peak:",sparkDict['time_to_peak'],"[Time Units?]")
        print("\tDistance to Outer Sarcolemma:",sparkDict['dist_to_sarcolemma'],"[microns]")
        print("\tSpark Amplitude:",sparkDict['spark_amplitude'],"[AU]")
        print("\tDuration Half Maximum:",sparkDict['duration_half_max'],"[Time Units?]")
        print("\tStandard Deviation of Fitted Gaussian:",sparkDict['sigma'])
    return sparks

def detectAndAnalyzeSparks_robust(fileName,
                                  thresholdValue=2.,
                                  temporalResolution=2.0, #[ms]/frame
                                  plotMe=False,
                                  verbose=False,
                                  readIntermediateFiles=True,
                                  writeIntermediateFiles=True,
                                  sparkTimeDilation=5,
                                  timeDilation=5, # frames to dilate spark detection by
                                  shortestTimeSpark=3 #frames
                                  ):

    ## Save the path to the files for later saving
    pathToFile = fileName.split('/')

    # ### 1. Read in Image and Clip as Needed
    #     

    ### Tifffile has some weird way of reading in lsm images in much higher dimension array than we need so we just select [0,0,:,:,:]
    if fileName[-4:] == '.lsm':
        img = tifffile.imread(fileName)[0,0,:,:,:]
    else:
        img = tifffile.imread(fileName).astype(np.float64)
        #print(np.shape(img))

        imgDims = np.shape(img)

    if not readIntermediateFiles:
        # ### 2. Smooth Image by Using Large Filter 
        ### The 120 size of the filter is ad hoc and could be subject to change
        ## Sparks are 20-40 ms duration half max so 120 should be large enough to not affect those transients
        import time
        start = time.time()
        largeKernel = np.ones((120,3,3),dtype=np.float)
        largeKernel /= np.sum(largeKernel)
        largeSmoothed = ndimage.convolve(img,largeKernel,mode='mirror')
        print("Total time to run large smoothing:", time.time()-start)

        # ### 3. Smooth Images using a Small Kernel to Denoise the Calcium Transients Without Smoothing them Out

        kernel = np.ones((3,5,5),dtype=np.float32)
        kernel /= np.sum(kernel)

        smallSmoothed = ndimage.convolve(img,kernel,mode='mirror')
    else:
        largeName = '/'.join(pathToFile[:-1])+'/largeSmoothed/'+pathToFile[-1]
        smallName = '/'.join(pathToFile[:-1])+'/smallSmoothed/'+pathToFile[-1]

        largeSmoothed = tifffile.imread(largeName)
        smallSmoothed = tifffile.imread(smallName)

    # ### INTERMEDIATE STEP:
    ### The distance to outer sarcolemma is very large in some cases, this is due to the routine not detecting 
    ###   that edges of the image are considered sarcolemma. To address this I believe we can pad the images
    ###   with zeros.
    paddedImage = np.zeros((imgDims[0],imgDims[1]+2,imgDims[2]+2),dtype=img.dtype)
    paddedLarge = np.zeros((imgDims[0],imgDims[1]+2,imgDims[2]+2),dtype=largeSmoothed.dtype)
    paddedSmall = np.zeros((imgDims[0],imgDims[1]+2,imgDims[2]+2),dtype=smallSmoothed.dtype)
    paddedImage[:,1:-1,1:-1] = img
    paddedLarge[:,1:-1,1:-1] = largeSmoothed
    paddedSmall[:,1:-1,1:-1] = smallSmoothed

    img = paddedImage
    largeSmoothed = paddedLarge
    smallSmoothed = paddedSmall

    imgDims = np.shape(img)

    # ### 4. Characterize Noise, Obtain Segmented Image of Cardiomyocyte, and Determine Points to Ignore

    noiseImg = img.copy() - largeSmoothed.copy()
    
    if writeIntermediateFiles:
        tifffile.imsave('/'.join(pathToFile[:-1])+'/noiseImages/'+pathToFile[-1],data=noiseImg.astype(np.int16))

    ### Getting the standard deviation of the noise
    noiseStd = np.std(noiseImg)

    ### Segmenting cardiomyocyte from the background based on noise characteristics
    #segmented = largeSmoothed > (2.75 * noiseStd + np.mean(noiseImg))
    segmented = smallSmoothed > (2.75 * noiseStd + np.mean(noiseImg))

    ### Perform morphological opening to deal with any noise specks that may have found their way into the data
    segmented = ndimage.morphology.binary_opening(segmented,
                                                  structure=np.ones((1,7,7)))
    
    ### Now we don't want to erode any of the columns where the sarcolemma is touching the border, so we mask this out
    segmentedColSum = np.sum(segmented.copy().astype(np.uint8),axis=1,keepdims=True)
    segmentationMask = segmentedColSum <= imgDims[1] - 5
    segmentationMask = np.broadcast_to(segmentationMask,imgDims)

    ### Perform morphological closing to close up any holes in the myocyte that may have arisen
    segmented = ndimage.morphology.binary_closing(segmented,
                                                  structure=np.ones((1,6,6)),
                                                  mask=segmentationMask)
    ### Perform morphological erosion to pull in the myocyte exterior, but we want to mask out the
    ###  border of the myocyte (sarcolemma). We find the columns where 

    segmented = ndimage.morphology.binary_erosion(segmented,
                                                  structure=np.ones((1,1,6),dtype=np.float),
                                                  mask=segmentationMask)
    
    ### Find Points that were masked in frame n but not masked in frame n+1, this gives an artifically high gradient
    ### We will mask these points out of the time gradient later
    falsePoints = segmented.copy().astype(np.uint8) - np.roll(segmented,-1,axis=0).astype(np.uint8) >= 1
    

    if writeIntermediateFiles:
        tifffile.imsave('/'.join(pathToFile[:-1])+'/segmentedImages/'+pathToFile[-1],data=segmented.astype(np.uint8))

    if plotMe:
        plt.figure()
        plt.imshow(segmented[100,:,:])
        plt.show()
    ### Perform some dilation to ensure we grab all of the myocyte?
    ## This didn't seem to be necessary previously

    if verbose:
        print("Noise Standard Deviation =",noiseStd)

    # ### 5. Divide Out the Long Time Average Image from the Short Time Average Image to Denoise the image

    transientImg = smallSmoothed.astype(np.float64) / largeSmoothed.astype(np.float64)

    ### Apply mask from segmented image
    #transientImg[segmented == 0] = 0

    if writeIntermediateFiles:
        tifffile.imsave('/'.join(pathToFile[:-1])+'/maskedTransients/'+pathToFile[-1],data=transientImg.astype(np.float32))

    # ### 6. Use Sobel Filter in the Time Direction to Pull Out Portions of the Image that Have High Gradients (Sparks)
    # Sparks seem to appear within a frame or two and are typically large intensity jumps. Should be able to pull this out really well with a 3D Sobel Filter
    timeGrad = ndimage.sobel(transientImg, axis=0, mode='nearest')
    timeGrad[segmented == 0] = 0
    timeGrad[falsePoints] = 0

    if writeIntermediateFiles:
        tifffile.imsave('/'.join(pathToFile[:-1])+'/timeGradients/'+pathToFile[-1],data=timeGrad.astype(np.float32))
    
    # ### 7. Threshold to Pull Out Sparks

    caSparks = timeGrad.copy() > thresholdValue

    ### time gradient > 8 or so is probably representative of contraction and not spark
    #caSparks[timeGrad > 5] = 0

    # ### 8. Perform Opening to Disregard Small Detections That are Likely Noise

    caSparks = ndimage.morphology.binary_opening(caSparks,
                                                 structure=np.ones((2,2,2)))


    # ### 9. Dilate in Time Dimension so We Get Better Context of Spark

    #caSparks = ndimage.morphology.binary_dilation(caSparks,
    #                                              structure=np.ones((5,1,1)))
    
    if writeIntermediateFiles:
        tifffile.imsave('/'.join(pathToFile[:-1])+'/sparkDetections/'+pathToFile[-1],data=caSparks.astype(np.uint8))

    # ### 10. Label and Delineate Segmented Sparks

    labeledSparks = measure.label(caSparks.copy(),background = 0)

    if verbose:
        print("Number of Detected Sparks:", np.max(labeledSparks))

    # ### 11. Pull Out Sparks, Average, and Plot Spark Characteristics

    ## Mask Transient Image with spark Detections
    maskedTransientImg = transientImg

    ## get region
    wholeRegion = measure.regionprops(labeledSparks)

    numSparkDetections = np.max(labeledSparks)
    #if numSparkDetections > 10:
    #    numSparkDetections = 10

    sparks = {}
    for sparkNum in range(numSparkDetections):
        ### Get region for spark number
        thisRegion = wholeRegion[sparkNum]

        ### Get coordinates for bounding box
        boundingCoordinates = np.asarray(thisRegion.bbox,dtype=np.uint64)
        boundingCoordinates = [int(coord) for coord in boundingCoordinates]
        
        if verbose:
            print("Bounding Coordinates", boundingCoordinates)
        ### Convert to easily referenced format
        sparkWindow = [[boundingCoordinates[0],boundingCoordinates[3]],
                       [boundingCoordinates[1],boundingCoordinates[4]],
                       [boundingCoordinates[2],boundingCoordinates[5]]]
        
        ### Get rid of noise by deleting sparks that are less than shortestTimeSpark
        if sparkWindow[0][1] - sparkWindow[0][0] < shortestTimeSpark:
            #print("Spark Number {} was deemed to be too short to be a spark. Deleting this spark".format(sparkNum))
            continue

        ### Dilate by specified spark time dilation 
        if sparkWindow[0][0] - timeDilation >= 0:
            #print type(sparkWindow[0][0])
            sparkWindow[0][0] -= timeDilation
        if sparkWindow[0][1] + timeDilation < imgDims[0]:
            sparkWindow[0][1] += timeDilation

        ### Create holder array that contains NaN where no spark was detected and transient spark where it was detected
        sparkHolder = maskedTransientImg.copy()[sparkWindow[0][0]:sparkWindow[0][1],
                                                sparkWindow[1][0]:sparkWindow[1][1],
                                                sparkWindow[2][0]:sparkWindow[2][1]]

        sparkValue_1D_avg = np.nanmean(np.nanmean(sparkHolder,axis=1),axis=1)
        ### Clip spark array to cutoff after max is reached
        sparkValue_1D_avg = sparkValue_1D_avg[:np.argmax(sparkValue_1D_avg)]
        x = np.arange(len(sparkValue_1D_avg))

        if plotMe:
            plt.figure()
            plt.title('Spark Characteristics for Spark #{}'.format(sparkNum))
            plt.ylabel('Spark Intensity [AU]')
            plt.xlabel('Time [ms]')
            plt.plot(x,sparkValue_1D_avg)
            plt.show()

        ### Store in dictionary for analyis if the 'spark' is short enough to be considered a spark and not a wave
        if sparkWindow[0][1] - sparkWindow[0][0] < 20:  
            sparks[str(sparkNum)] = {'spark_array':sparkValue_1D_avg,
                                 'spark_coordinates':sparkWindow}
    

    # ### 12. Fit Exponential to Data to Find Time of Half Maximum and Time to Half Maximum
    sparksToDelete = []
    for sparkNum, sparkDict in sparks.items():
        sparkArray = sparkDict['spark_array']
        x = np.arange(len(sparkArray))

        ### Consider clipping sparkArray to max value
    
        fitFunction = lambda t,b,c: np.min(sparkArray) + b* np.exp(c*t)
        #fitFunction = lambda t,a,b,c: a + b* np.exp(c*t)
        try:
            coeffs =  scipy.optimize.curve_fit(fitFunction, x, sparkArray,maxfev = 3000)
        except:
            print("Coefficients not found for spark #{}, continuing without analyzing this spark".format(sparkNum))
            sparksToDelete.append(sparkNum)
            continue

    
        fitFunctionValues = fitFunction(x,coeffs[0][0],coeffs[0][1])
    
        if plotMe:
            plt.figure()
            plt.plot(x,sparkArray,label='Experimental')
            plt.plot(x,fitFunctionValues,label='Fit')
            plt.legend()
            plt.show()
    
        halfMax = (np.max(fitFunctionValues) + np.min(fitFunctionValues)) / 2.
        timeHalfMax = np.argmin( np.sqrt((sparkArray - halfMax)**2 )) - np.argmin(sparkArray)
        sparkAmplitude = np.max(sparkArray) / np.mean(sparkArray[:3])
        timeToPeak = len(sparkArray) - np.argmin(sparkArray)
        if verbose:
            print("Spark #{}".format(sparkNum))
            print("\tHalf Maximum Value:", halfMax)
            print("\tTime to Half Maximum:", timeHalfMax)
            print("\tSpark Amplitude:",sparkAmplitude)
            print("\tTime to Peak:",timeToPeak)
            print()
    
        sparkDict['half_maximum'] = halfMax
        sparkDict['time_to_half_maximum'] = timeHalfMax
        sparkDict['spark_amplitude'] = sparkAmplitude
        sparkDict['time_to_peak'] = timeToPeak

    ## Delete the bad sparks
    if np.shape(sparksToDelete)[0] != 0:
        for sparkNum in sparksToDelete:
            del sparks[sparkNum]

    # ### 13. Find Centroid of Spark so We Can Analyze How Close it is to Sarcolemmal Surfaces
    ### We need to generate a holder that has the distance calculated for each point in the original segmented image
    seg = segmented[0,:,:].copy()
    seg = seg.astype(np.uint8)
    distanceArray = ndimage.morphology.distance_transform_edt(seg)
    if plotMe:
        plt.figure()
        plt.imshow(distanceArray)
        plt.show()
    for sparkNum, sparkDict in sparks.items():
        thisRegion = wholeRegion[int(sparkNum)]
        cx, cy = thisRegion.centroid[1:]
        cx, cy = int(round(cx)), int(round(cy))
        centroid = cx, cy
        dist = distanceArray[cx,cy]
        dist = convToUm(dist)
        if verbose:    
            print("Spark Number {} Centroid:".format(sparkNum),centroid)
            print("Distance to Sarcolemmal Surface:", dist)
        sparkDict['centroid'] = centroid
        sparkDict['dist_to_sarcolemma'] = dist


    # ### 14. Find the Time at Half Maximum and delete sparks that are too long to be considered sparks (likely waves at this point)
    # This could be really tricky. Almost like we'll need to apply a region growing method for the time gradient 
    
    sparksToDelete = []
    for sparkNum, sparkDict in sparks.items():
        ### Pull out the final time we decided for the spark peak
        coords = sparkDict['spark_coordinates']
        finalSparkTime = coords[0][1]
    
        condition = True
        decayedSparkTime = finalSparkTime
        while condition:
            sparkValue = np.mean(transientImg[int(decayedSparkTime),
                                                coords[1][0]:coords[1][1],
                                                coords[2][0]:coords[2][1]])
    
            ### Check that region growing condition is still satisfied
            ## We want to continue growing the region as long as the current time gradient and this time gradient is negative
            # SUBJECT TO CHANGE
            condition = sparkValue > sparkDict['half_maximum']
            decayedSparkTime += 1
    
            ### Check if decayedSparkTime is still within image bounds
            if decayedSparkTime == imgDims[0]:
                decayedSparkTime -= 1
                break
        
        ### Mark 'spark' for deletion if half max time is too long
        thisFullSpark = transientImg.copy()[coords[0][0]:int(decayedSparkTime),
                                                coords[1][0]:coords[1][1],
                                                coords[2][0]:coords[2][1]]
        fullSpark1D = np.mean(np.mean(thisFullSpark,axis=1),axis=1)
        duration_half_max = len(fullSpark1D) - sparkDict['time_to_half_maximum']
        if duration_half_max > 50. / temporalResolution:
            sparksToDelete.append(sparkNum)
        else:
            if verbose:
                print(decayedSparkTime)
    
    
    
            if plotMe:
                plt.figure()
                plt.plot(np.arange(len(fullSpark1D)),fullSpark1D)
                plt.show()
    
            sparkDict['decay_to_half_max'] = len(fullSpark1D)
            sparkDict['duration_half_max'] = duration_half_max
    
    if verbose:
        print("Deleting these sparks since they are too long to be considered sparks:",sparksToDelete)
    ### Delete sparks that are too long to be considered sparks
    for spark in sparksToDelete:
        del sparks[spark]
    
    # ### 15. Analyze Full Width Half Maximum
    
    for sparkName, sparkDict in sparks.items():
        coords = np.asarray(sparkDict['spark_coordinates'])
        timeAtHalfMaximum = coords[0][0] + sparkDict['time_to_half_maximum']
        if verbose:
            print(coords)
        ## try fitting half maximum window to be a little bigger so the gaussian fit is more accurate
        #coords += 5
        wideningParameter = 5
        if coords[1][0] > wideningParameter:
            coords[1][0] -= wideningParameter
        if coords[1][1] + wideningParameter < imgDims[1]:
            coords[1][1] += wideningParameter
        if coords[2][0] > wideningParameter:
            coords[2][0] -=wideningParameter
        if coords[2][1] + wideningParameter < imgDims[2]:
            coords[2][1] += wideningParameter
    
        if verbose:
            print(coords)
    
        halfMaximumWindow = transientImg[int(timeAtHalfMaximum),
                                         int(coords[1][0]):int(coords[1][1]),
                                         int(coords[2][0]):int(coords[2][1])]
        shape = np.shape(halfMaximumWindow)
        x_0,y_0 = shape
        x = np.arange(x_0)
        y = np.arange(y_0)
    
        x,y = np.meshgrid(x,y)
    
        ### Fit gaussian to half maximum time
        maxPoint_row,maxPoint_col = np.unravel_index(np.argmax(halfMaximumWindow),shape)
        initial_guess = (np.max(halfMaximumWindow), 3, 0, 1)
        params, _ = scipy.optimize.curve_fit(twoD_Gaussian, (x,y,maxPoint_row,maxPoint_col), halfMaximumWindow.flatten(),p0=initial_guess)
    
        if verbose:
            print("Sigma:", convToUm(params[1]))
    
        #data_fitted = twoD_Gaussian((x,y,maxPoint_row,maxPoint_col), *params)
    
        if plotMe:
            fig,ax = plt.subplots(1,1)
            ax.imshow(halfMaximumWindow, origin='bottom')
            plt.show()
    
    
        sparkDict['sigma'] = convToUm(params[1])
    
    # ### 16. Display All Parameters of Interest and convert to proper units
    counter = 0
    for sparkNum, sparkDict in sparks.items():
        counter += 1
        print("Spark Number:", sparkNum)
        #for quality, quantity in sparkDict.items():
        #    print"\t"+quality+":",quantity
        sparkDict['time_to_peak'] *= temporalResolution
        print("\tTime to Peak:",sparkDict['time_to_peak'],"[ms]")
        print("\tDistance to Outer Sarcolemma:",sparkDict['dist_to_sarcolemma'],"[microns]")
        print("\tSpark Amplitude:",sparkDict['spark_amplitude'],"[AU]")
        sparkDict['duration_half_max'] *= temporalResolution
        print("\tDuration Half Maximum:",sparkDict['duration_half_max'],"[ms]")
        print("\tStandard Deviation of Fitted Gaussian:",sparkDict['sigma'],"[um]")
    print("Final Spark Count:",counter)
    
    return sparks

def smoothRawDataAndSave(fileName):
  '''
  Function to chunk up the computational expense of performing the smoothing with the massive kernels.
    This way we can save the intermediate results and seriously cut down on the time needed to rerun
    analysis.
    
  Input:
    fileName -> String pointing to the path of the raw image

  Output:
    Saves fileName_largeSmoothed.tif and fileName_smallSmoothed.tif indicating the smoothing using the large kernel and the small kernel
  '''
  # ### 1. Read in image
  if '.lsm' in fileName:
    img = tifffile.imread(fileName)[0,0,:,:,:]
  else:
    img = tifffile.imread(fileName)

  # ### 2. Smooth Image by Using Large Filter 
  ### The 120 size of the filter is ad hoc and could be subject to change
  ## Sparks are 20-40 ms duration half max so 120 should be large enough to not affect those transients
  import time
  start = time.time()
  largeKernel = np.ones((120,3,3),dtype=np.float)
  largeKernel /= np.sum(largeKernel)
  largeSmoothed = ndimage.convolve(img,largeKernel,mode='mirror')
  print("Total time to run large smoothing:", time.time()-start)

  # ### 3. Smooth Images using a Small Kernel to Denoise the Calcium Transients Without Smoothing them Out

  kernel = np.ones((3,5,5),dtype=np.float32)
  kernel /= np.sum(kernel)

  smallSmoothed = ndimage.convolve(img,kernel,mode='mirror')

  ### testing the saving of this image
  pathList = fileName.split('/')
  largeName = '/'.join(pathList[:-1])+'/largeSmoothed/'+pathList[-1]
  smallName = '/'.join(pathList[:-1])+'/smallSmoothed/'+pathList[-1]

  tifffile.imsave(largeName,data=largeSmoothed)
  tifffile.imsave(smallName,data=smallSmoothed)




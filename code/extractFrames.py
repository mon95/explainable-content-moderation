import cv2 as cv
import os
import os.path

class FrameExtractor:

    @staticmethod
    def extractFramesForVideo(fileName, frameRate, outputName, outputDir):
        """

            Extracts teh frames in the video and writes teh image to the outputDir

            Input:
            fileName; video file
            outputDir: directory to which the images are extracted
            frameRate: the interval at which teh frame is extracted, i.e at every 2 seconds, every 0.5 secons
            outputName: fileName for the image

        """


        video = cv.VideoCapture(fileName)
        count = 1 # Keep track of the number of frames extracted and also to append as a marker to the image
        while (video.isOpened()):
            hasFrame, frame = video.read()
            if not hasFrame:
                video.release()
                break
            outputPath = os.path.join(outputDir, outputName)
            cv.imwrite(outputPath+str(count)+'.jpg', frame)
            video.set(cv.CAP_PROP_POS_MSEC, count * frameRate*1000) # in millisecs so * 1000
            count += 1

    @staticmethod
    def extract(dirLocation, outputDirLocation, relativePath=False,frameRate=0.5, recursive=False):
        """

            Traverses the directory = dirLocation and extracts frames from all files in the directory

            Input:
            dirLocation; directory that contains all video files
            outputDirLocation: directory to which the images are extracted
            relativePath: make life easier by giving relative path from the script
            frameRate: the interval at which teh frame is extracted, i.e at every 2 seconds, every 0.5 secons
            recursive: if you want to recursivwly search for files in the folder

        """
        
        
        
        frameRate = 0.5 #default is one frame every 0.5 seconds
        # create ouptup dir if it does not exist
        if relativePath:
            dirName = os.path.dirname(__file__)
            dirLocation = os.path.join(dirName, dirLocation)
            outputDirLocation = os.path.join(dirName, outputDirLocation)

        if not os.path.isdir(outputDirLocation):
            os.mkdir(outputDirLocation)

        for dirPath, dirNames, fileNames in os.walk(dirLocation):
            for fileName in fileNames:
                outputName = fileName
                fileName = os.path.join(dirLocation, fileName)
                FrameExtractor.extractFramesForVideo(fileName, frameRate, outputName, outputDirLocation)
            if not recursive:
                break


if __name__ == '__main__':
    # You can gvie relative paths
    dirLocation = 'Violence'
    outputDirLocation = 'outputViolenceFolder'
    FrameExtractor.extract(dirLocation, outputDirLocation, relativePath=True)

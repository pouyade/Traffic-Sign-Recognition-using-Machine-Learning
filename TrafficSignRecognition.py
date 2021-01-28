import tensorflow as tf
import cv2
from Config import Config
import Utilities
from sklearn.externals import joblib
import numpy as np
from Detection.Drawer import Drawer
from Detection.PreProccessor import PreProccessor
from Detection.SignDetector import SignDetector
import time

class TrafficSignRecogniser(object):

    orginalImage = None
    editedImage = None
    editedImages = []
    NormalizedImage  = None
    drawedImage = None
    ImgCnnArray = []
    predictRes = []
    singCaptions = []
    all_scores = []
    singClasses = []
    DetectionModel = None
    detecteds_location = []
    detecteds_images = []
    write_on_detections = True
    def __init__(self):
        self.DetectionModel = joblib.load(Config.DETECTION_MODEL_PATH)
        self.RecognitionModel = tf.keras.models.load_model(Config.RECOGNITION_MODEL_PATH)
        Utilities.loadSignImages()

    def PreProccess(self):
        self.editedImage = self.orginalImage.copy()
        # self.editedImage = PreProccessor.Clahe(self.orginalImage)
        self.NormalizedImage = PreProccessor.Normalize(self.editedImage)
        self.NormalizedImage2 = PreProccessor.Normalize2(self.editedImage)
        # cv2.imwrite("/Users/pouyadark/Desktop/test/normal2.jpg", self.NormalizedImage2)

        self.editedImage = PreProccessor.BlurImage(self.editedImage)
        self.NormalizedImage = PreProccessor.BlurImage(self.NormalizedImage)
        self.NormalizedImage2 = PreProccessor.BlurImage(self.NormalizedImage2)

        self.editedImages = Utilities.applyMask(self.editedImage,self.NormalizedImage,self.NormalizedImage2, Config.MASKS)
        # Utilities.ShowImgArray( self.editedImages)
        # self.editedImages = Utilities.MulticvtColor(self.editedImages,cv2.COLOR_BGR2GRAY)
        # self.editedImage = PreProccessor.RemoveSmallObjects(self.editedImage)
        # self.editedImages = PreProccessor.Thresholding(self.editedImages)
        return self.editedImages

    def DetectSigns(self):
        self.detecteds_images,self.detecteds_location,self.all_scores,self.drawedImage = SignDetector.detect(self.orginalImage,self.editedImages,self.DetectionModel)
        return

    def Recognition(self):
        self.singCaptions = []
        self.singClasses = []
        self.ImgCnnArray = Utilities.filterArrayimageSings(self.detecteds_images)
        if (len(self.ImgCnnArray) == 0):
            return
        imgs = [self.ImgCnnArray]
        self.predictRes = self.RecognitionModel.predict(imgs)
        for i in range(len(self.predictRes)):
            classid=Utilities.biggestIndex(self.predictRes[i])
            confedece=Utilities.biggest(self.predictRes[i])
            detectscore=self.all_scores[i]
            if(confedece >= Config.RECOGNITION_MIN_CONFODENCE):
                caption = Config.SIGN_LABELS[classid]
                self.singCaptions.append(caption)
                self.singClasses.append(classid)
                if self.write_on_detections:
                    loc = self.detecteds_location[i]
                    self.drawedImage = Drawer.DrawSigns(self.drawedImage,loc,caption,classid,confedece,detectscore)
        return

    def RunImage(self,image,showtime = True):
        if(Config.INPUT_RESIZE):
            width = Config.INPUT_IMAGE_WIDTH
            height =int(width * (float(image.shape[0])/float(image.shape[1])))
            self.orginalImage = cv2.resize(image.copy(),(width,height))
        else:
            self.orginalImage = image.copy()

        self.Run(showtime)
        return self.drawedImage

    def Run(self,showtime = True):
        start_time = time.time()
        self.PreProccess()
        self.DetectSigns()
        self.Recognition()
        if (showtime):
            print("--- %s seconds ---" % (time.time() - start_time))
        return

    def TrainDatasetTest(self):
        self.testData(Config.DETECT_TRAIN_DATA_SET_PATH)

    def FullDatasetTest(self):
        self.testData(Config.FULL_DATASET_PATH)

    def CreateDetectionDataset(self,path):
        print("Creating Test Detection Dataset...")
        detection_dataset = Utilities.readDetectionDataset(path)
        print("Compeleted Dataset.")
        print("Saving Dataset...")
        Utilities.save_array_to_file(detection_dataset, Config.PICKLES_PATH +"detection_dataset.pickle")
        print("Dataset Saved.")

    def testData(self,mainpath):
        self.CreateDetectionDataset(mainpath)
        print("Load Train Detection Dataset...")
        datasetdict = Utilities.read_array_from_file(Config.PICKLES_PATH + "detection_dataset.pickle")
        print("Dataset Loaded.")
        print("Start Testing...")
        start_time = time.time()
        count = 0
        correct = 0
        wrong = 0
        countofsigns = 0
        countofsignsdetected = 0
        partialcorrect = 0
        resclasses = np.zeros((43,2))

        Utilities.DeleteAllFilesInFolder(Config.CORRECT_RESTUL_PATH)
        Utilities.DeleteAllFilesInFolder(Config.PARTIAL_RESTUL_PATH)
        Utilities.DeleteAllFilesInFolder(Config.WRONG_RESTUL_PATH)
        Utilities.DeleteAllFilesInFolder(Config.SIGN_POSTIVE_RESTUL_PATH)
        Utilities.DeleteAllFilesInFolder(Config.SIGN_NEGETIVE_RESTUL_PATH)

        for filename, classarray in datasetdict.items():
            count +=1
            countofsigns += len(classarray)
            Utilities.Progressbar(count,len(datasetdict),start_time,correct,partialcorrect,wrong)
            Utilities.Log("TrainTest","Image #" + str(count))
            path = mainpath + filename
            img = cv2.imread(path,1)
            self.RunImage(img,False)
            countofsignsdetected += Utilities.DetectCorrectSignCount(self.singClasses,classarray)
            resclasses = Utilities.CountClassess(resclasses,self.singClasses,classarray)
            Utilities.Log("TrainTest","Detected:")
            Utilities.Log("TrainTest",self.singClasses)
            Utilities.Log("TrainTest","Correct Answer: ")
            Utilities.Log("TrainTest",classarray)

            if(sorted(self.singClasses) == sorted(classarray)):
                correct+=1
                Utilities.Log("TrainTest",filename + " is Correct")
                cv2.imwrite(Config.CORRECT_RESTUL_PATH + filename + ".jpg",self.drawedImage)
            else:
                if(Utilities.searchInArray(self.singClasses,classarray)):
                    partialcorrect +=1
                    Utilities.Log("TrainTest",filename + " is Partial Correct")
                    cv2.imwrite( Config.PARTIAL_RESTUL_PATH + filename + ".jpg",self.drawedImage)
                else:
                    wrong+=1
                    Utilities.Log("TrainTest",filename + " is Wrong")
                    cv2.imwrite(Config.WRONG_RESTUL_PATH + filename + ".jpg",self.drawedImage)
        Utilities.PrintResult(count,countofsigns,countofsignsdetected,correct,partialcorrect,wrong,start_time,resclasses)

    def testTestData(self,image):
        self.orginalImage = image.copy()
        self.Run()
        return self.drawedImage

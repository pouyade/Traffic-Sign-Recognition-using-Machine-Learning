import cv2
import os
class Config:

#MASK [LOWER,HIGHER,(img=0,norml1=1,normal2=2)]

    MASKS = {
        'red':[
            [[0,50,50],[10,255,255], 0],#Red #5 Good
            [[170,50,50],[180,255,255], 0],#Red #5 Good

            [[148, 77, 76], [180, 141, 151], 1],  # Red #5
            [[162, 42, 93], [174, 59, 104], 1],  # Red #5
            [[3, 120, 111], [14, 190, 143], 1],  # Red #5

            [[148, 88, 97], [254, 174, 174], 1],  # Red #5


        ],
        'red2':[
            [[0, 146, 0], [230, 178, 255], 1],  # Red #5import
            [[149, 90, 6], [255, 255, 184], 1],  # Red #5
            [[0, 41, 118], [6, 143, 136], 1],  # Red #5
            [[1, 168, 158], [16, 255, 255], 1],  # Red #5

            [[138, 100, 201], [195, 160, 228], 2],  # Red #5

        ],
        'red3':[
            [[0, 67, 34], [16, 135, 57], 0],  # Red #5

            # [[0, 124, 199], [12, 160, 217], 1],  # Red #5
            # [[0, 99, 204], [9, 189, 255], 2],  # Red #5


        ],
        # 'red4':[
        #         [[151, 20, 79],  [184, 59, 121], 0],  # Red #5
            # ],
        'blue':[
            [[98,107,177], [114,255,207], 1],  # Red #5
            [[97, 72, 100], [124, 148, 128], 1],  # Red #52
            [[109, 133, 60], [122, 190, 145], 1],  # Red #5

            [[88, 167, 217], [114, 205, 243], 2],  # Red #5


        ],
        'blue2':[
            [[110, 169, 217], [124, 203, 227], 1],  # Red #5
            [[101, 200, 231], [114, 232, 238], 2],  # Red #5
        ],
        'blue3': [
            [[75, 30, 226], [108, 255, 252], 0],  # Red #5
        ],
        # 'orange': [
        #      # [[29,44,2],[30,112,255],2],
        #      # [[28,86,255],[34,144,255],2],
        #      [[11,165,203],[17,192,214],2],
        # ],
        'orange2':[
            # [[4, 3, 78], [32, 74, 102], 1],
            [[11, 172, 193], [16, 205, 254], 2],
            # [[18, 115, 176], [22, 141, 189], 2],
            [[14, 170, 194], [25, 187, 246], 2],

            [[29, 72, 113], [33, 130, 175], 2],
        ],


    }
    #For noise Removal
    THRESHOLD_MIN = 50
    THRESHOLD_MAX = 200
    MIN_OBJECT_SIZE = 400
    # MIN_OBJECT_SIZE = 350
    CLAHE_GRID_SIZE = 5
    SINGLE_ROI_RATIO = (0.75,1.2)
    Duble_RIO_RATIO = (0.45, 0.75)
    Trible_RIO_RATIO = (0.3, 0.45)
    INPUT_RESIZE = False
    INPUT_IMAGE_WIDTH = 1200

    #Detection
    RECOGNITION_MIN_CONFODENCE = 0.35
    WINDOW_SIZE_WIDTH = 64
    WINDOW_SIZE_HEIGHT = 64
    WINDOW_SIZE = (WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT)
    WINDOW_SLIDE_STEP = 0.05
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    threshold = 0.55
    downscale = 1.5
    MULTI_DETECT_THRESHOLD_RANGE = (-15,15)

    #DEBUG
    DEBUG_DRAW_ROIS = False
    DEBUG_DRAW_MERGED_ROIS = False
    DEBUG_DRAW_REMOVED_ROIS = False
    DEBUG_DRAW_SCORES = True
    DEBUG_PRINT_ERRORS = False
    DEBUG_SAVE_SIGNS = False
    DEBUG_DRAW_DETECT_REMOVED_ROIS = False


    #Graphic
    PROGRESS_BAR_SIZE = 60
    PROGRESS_BAR_FILLED_CHAR = "-"
    PROGRESS_BAR_UNFILLED_CHAR = " "
    PROGRESS_BAR_CURRENT_CHAR = ">"
    #text
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_fontScale = 0.8
    TEXT_fontColor = (255, 0, 0)
    TEXT_lineType = 2


    #Paths

    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    PATH_SEPERATOR = os.path.sep
    DATA_SET_PATH = CURRENT_PATH + PATH_SEPERATOR + "Dataset" + PATH_SEPERATOR
    FULL_DATASET_PATH = DATA_SET_PATH  +'FullIJCNN2013' + PATH_SEPERATOR
    MODELS_PATH = CURRENT_PATH + PATH_SEPERATOR +'Model' + PATH_SEPERATOR
    SIGNS_PATH = CURRENT_PATH + PATH_SEPERATOR +'Signs' + PATH_SEPERATOR
    LOGS_PATH = CURRENT_PATH + PATH_SEPERATOR +'Logs' + PATH_SEPERATOR
    PICKLES_PATH = CURRENT_PATH + PATH_SEPERATOR +'Pickles' + PATH_SEPERATOR

    DETECT_TRAIN_DATA_SET_PATH = DATA_SET_PATH + "TrainIJCNN2013" + PATH_SEPERATOR
    POUYA_DATA_SET_PATH = DATA_SET_PATH + "PouyaDataset" + PATH_SEPERATOR
    DETECT_TEST_DATA_SET_PATH = DATA_SET_PATH + "TestIJCNN2013" + PATH_SEPERATOR
    RECOGNITION_TRAIN_DATA_SET_PATH = DATA_SET_PATH + "GTSB" + PATH_SEPERATOR
    RECOGNITION_TEST_DATA_SET_PATH = DATA_SET_PATH + "GTSB_Test" + PATH_SEPERATOR
    MY_PHOTOS_PATH = CURRENT_PATH + PATH_SEPERATOR + 'MyPhotos'+ PATH_SEPERATOR
    DETECTION_MODEL_PATH = MODELS_PATH + "detection_model.npy"
    RECOGNITION_MODEL_PATH = MODELS_PATH + "traffic.model"

    RESTUL_PATH = CURRENT_PATH + PATH_SEPERATOR +'ResultImages' + PATH_SEPERATOR
    CORRECT_RESTUL_PATH = RESTUL_PATH +'Correct' + PATH_SEPERATOR
    WRONG_RESTUL_PATH = RESTUL_PATH  +'Partial' + PATH_SEPERATOR
    PARTIAL_RESTUL_PATH = RESTUL_PATH  +'Wrong' + PATH_SEPERATOR
    SIGN_RESTUL_PATH = RESTUL_PATH  +'Signs' + PATH_SEPERATOR
    SIGN_POSTIVE_RESTUL_PATH = SIGN_RESTUL_PATH + 'Postive'+ PATH_SEPERATOR
    SIGN_NEGETIVE_RESTUL_PATH = SIGN_RESTUL_PATH + 'Negetive'+ PATH_SEPERATOR

    CHECK_HOG = True


    DETECT_DATA_SET_GUID_CSV = "gt.txt"

    # IMAGE_SIZE = 64
    IMAGE_SIZE = 150
    EPOCHS_COUNT = 1000
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 512
    SIGN_IMAGES = []
    SIGN_GROUPS = {
        'prohibitory':[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16],
        'mandatory': [33, 34, 35, 36, 37, 38, 39, 40],
        'danger': [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    }
    SIGN_LABELS = [
        "Speed_20",
        "Speed_30",
        "Speed_50",
        "Speed_60",
        "Speed_70",
        "Speed_80",
        "Unlimit_80",
        "Speed_100",
        "Speed_120",
        "No_overTaking",
        "No_overTaking_for_Trucks",
        "Subway_and_Main_intersection",
        "Main_road",
        "Junction_ahead",
        "Stop",
        "No_entrance_from_both_end",
        "No_entrance_for_Trucks",
        "No_entrance",
        "Danger",
        "Left_turn",
        "Right_turn",
        "Successive_bolts_first_left",
        "speed_bump",
        "slip_road",
        "Narrow_Road",
        "under_repair",
        "traffic lights",
        "People_Passage",
        "Children_Passage",
        "no_entrance_for_bicycle",
        "Snow",
        "Wild_Animals_passage",
        "No_limits",
        "blue_Right_turn",
        "blue_Left_turn",
        "Straight_forward",
        "can_go_forward_or_right",
        "can_go_forward_or_left",
        "just_right",
        "just_left",
        "square",
        "no_limit_for_overtake",
        "no_limit_for_overtake_truk",
        "Side_Walk"
    ]

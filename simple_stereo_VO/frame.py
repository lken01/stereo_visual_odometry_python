import cv2
import numpy as np 


class Frame:

    def __init__(self, ID, imgL, imgR, intrinsics, dist_coeffs):
        self.frame_ID = ID
        self.imgL = imgL
        self.imgR = imgR
        self.intrinsics = intrinsics


    def extract_features(self, ftype = "orb"):
        if ftype.lower() == "orb":
            detector = cv2.ORB_create(1000)
            extractor = detector
        else:
            # TO DO: define other types of features extractors
            print("Invalid feature extractor"); sys.exit()

        self.kpl, self.desl = self.feature_extractor(self.imgL,
                                            detector, extractor)
        self.kpr, self.desr = self.feature_extractor(self.imgR,
                                            detector, extractor)


    @staticmethod
    def feature_extractor(image, detector, extractor):
        keypoints = detector.detect(image)
        descriptors = extractor.compute(image, keypoints)
        return keypoints, descriptors[1]


    def get_measurements(self):
        self.match_features("brute_force")
        self.traingulate_points()


    def match_features(self,mtype="brute_force"):
        self.matches = []
        if mtype.lower() == "brute_force":
            bfm = cv2.BFMatcher()
            bfMatches = bfm.knnMatch(self.desl, self.desr, k=2)
            for m,n in bfMatches:
                if m.distance < 0.75*n.distance:
                    self.matches.append(m)

        elif mtype.lower() == "flann":
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary

            desl = np.float32(self.desl)
            desr = np.float32(self.desr)
            
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(desl,desr,k=2)

            for i,(m,n) in enumerate(matches):
                if m.distance < 0.75*n.distance:
                    self.matches.append(m)

        else:
            print("Invalid matching method. Use 'flann' or 'brute_force')")
            sys.exit()


    def traingulate_points(self):
        self.convertMatchP2xy()

        fx = self.intrinsics[0]; fy = self.intrinsics[1]
        cx = self.intrinsics[2]; cy = self.intrinsics[3]
        baseline = self.intrinsics[6]

        self.points3d = []

        for lp,rp in zip(self.lpts, self.rpts):
            lp = np.array(lp)
            rp = np.array(rp)
            disparity = lp[0] - rp[0]

            if disparity > 0:
                depth = (fx*baseline)/disparity
                x = ((lp[0]-cx)*depth)/fx
                y = ((lp[1]-cy)*depth)/fy
                self.points3d.append([x,y,depth])
            else:
                self.points3d.append([0,0,0])


    def convertMatchP2xy(self):
        self.lpts, self.rpts = [], []
        self.mkpl, self.mkpr = [], []
        self.mdesl, self.mdesr = [], []
        for idx in range(len(self.matches)):
            # (u,v) points
            self.lpts.append(self.kpl[self.matches[idx].queryIdx].pt)
            self.rpts.append(self.kpr[self.matches[idx].trainIdx].pt)
            # Keypoints
            self.mkpl.append(self.kpl[self.matches[idx].queryIdx])
            self.mkpr.append(self.kpr[self.matches[idx].trainIdx])
            # Descriptors
            self.mdesl.append(self.desl[self.matches[idx].queryIdx])
            self.mdesr.append(self.desr[self.matches[idx].trainIdx])

        assert len(self.lpts) == len(self.rpts),\
        "Mismatch left and right point count"


    def transform_3Dpoints(self,pose):
        R = np.array(pose["rmat"])
        t = np.array(pose["tvec"])
        transformed_points = np.array([])
        for idx in range(len(self.points3d)):
            if self.points3d[idx][0] is not None:
                transformed_points = np.append(transformed_points,\
                                        R.T.dot(self.points3d[idx])\
                                        +(-R.T.dot(t)))
        self.points3d = transformed_points.reshape(
                                int(len(transformed_points)/3),3).tolist()

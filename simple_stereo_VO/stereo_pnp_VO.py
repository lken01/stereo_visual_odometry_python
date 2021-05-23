import cv2
import numpy as np 
from glob import glob
import sys
import time
import os

from frame import Frame
from tracker import Track


class Stereo_PnPVO:

    def __init__(self, intrinsics, dist_coeffs):
        self.poses = {}
        self.measurements = {}
        self.point_data = {}
        self.keyframe_ID = 0
        self.prev_trackedPts2D = None
        self.prev_trackedPts3D = None

        K_mat = np.array([[intrinsics[0], 0., intrinsics[2]],
                          [0., intrinsics[1], intrinsics[3]],
                          [0., 0., 1.]])
        self.cam_params = {"intrinsics":intrinsics,
                           "camera_matrix":K_mat,
                           "dist_coeffs":dist_coeffs}

        # Percentage of keyframe points to decide new keyframe
        self.min_pt4kf = 0.7


    def initialize(self, frame):
        is_keyframe = True
        self.update_keyframeID()

        frame.get_measurements()
        rotation_matrix = np.eye(3)
        translation_vec = np.zeros(3)
        pose = np.hstack((rotation_matrix,translation_vec.reshape(3,1)))
        self.poses[frame.frame_ID] = {"frame_ID":frame.frame_ID,
                                      "pose":pose.tolist(),
                                      "rmat":rotation_matrix.tolist(),
                                      "tvec":translation_vec.tolist(),
                                      "position":translation_vec.tolist()}
        self.update_measurements(frame)
        self.point_data[frame.frame_ID] = {"frame_ID":frame.frame_ID,
                                           "points2d":frame.lpts,
                                           "points3d":frame.points3d}


    def track(self, frame, prev_imgL, prev_imgR):
        tracker = Track(frame, prev_imgL, prev_imgR, self.prev_trackedPts2D)
        tracker.track_features(self.measurements[self.keyframe_ID]["points3d"],
                               self.prev_ptIDs)

        print("tracked points {} of {}".format(
                    len(self.measurements[self.keyframe_ID]["points2dL"]),
                    len(tracker.tracked_pts)))

        # Camera pose estimation
        self.pnp_solver(tracker.tracked_pts, tracker.corres_3Dpts)

        # Required for data association
        self.prev_trackedPts2D = tracker.tracked_pts
        self.prev_ptIDs = tracker.point_IDs

        # Store points used
        self.point_data[frame.frame_ID] = {"frame_ID":frame.frame_ID,
                                           "points2d":tracker.tracked_pts,
                                           "points3d":tracker.corres_3Dpts}

        # Check if new keyframe should be defined
        new_kfFactor = len(self.measurements[self.keyframe_ID]["points2dL"])\
                        * self.min_pt4kf

        if len(tracker.tracked_pts) < new_kfFactor:
            print("new keyframe at {}. kf_ID: {}".format(\
                                    frame.frame_ID,self.keyframe_ID))
            is_keyframe = True
            self.update_keyframeID()
            frame.get_measurements()
            frame.transform_3Dpoints(self.poses[frame.frame_ID])
            self.update_measurements(frame)


    def pnp_solver(self, points2d, points3d):
        points2d = np.array(points2d)
        points3d = np.array(points3d)
        intrinsic_matrix = self.cam_params["camera_matrix"]
        dist_coeffs = self.cam_params["dist_coeffs"]

        try:
            #flags: cv2.SOLVEPNP_P3P; cv2.SOLVEPNP_EPNP; cv2.SOLVEPNP_ITERATIVE
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                                                    points3d,points2d,
                                                    intrinsic_matrix,
                                                    dist_coeffs,
                                                    flags=cv2.SOLVEPNP_EPNP,
                                                    reprojectionError=2,
                                                    iterationsCount=100)
        except Exception as e:
            print("\n-->PnP failed possibly due to less number of points\n")
            print(e); sys.exit()


        if success:
            rmat, Jacobian = cv2.Rodrigues(rvec)
            pose = cv2.hconcat((rmat, tvec))

            rotation_matrix = pose[:3,:3]
            translation_vec = pose[:3,3]
            cur_position = -rotation_matrix.T.dot(translation_vec)
            print("position:",cur_position)

            self.poses[frame.frame_ID] = {"frame_ID":frame.frame_ID,
                                          "pose":pose.tolist(),
                                          "rmat":rotation_matrix.tolist(),
                                          "tvec":translation_vec.tolist(),
                                          "position":cur_position.tolist()}
        else:
            print("failed to estimate pose!"); sys.exit()


    def update_measurements(self,frame):
        self.measurements[self.keyframe_ID] = {"frame_ID":frame.frame_ID,
                                               "points2dL":frame.lpts,
                                               "points2dR":frame.rpts,
                                               "points3d":frame.points3d}
        self.prev_trackedPts2D = frame.lpts
        self.prev_ptIDs = [i for i in range(len(frame.lpts))]


    def update_keyframeID(self):
        self.keyframe_ID += 1


    def plot_trajectory(self, sframe,end_frame, plot3D=False, plot_points = False):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        #camera Y-axis is vertical axis
        xc,yc,zc = [],[],[]
        for idx in range(sframe,end_frame):
            position = self.poses[idx]["position"]
            xc.append(position[0])
            yc.append(position[1])
            zc.append(position[2])

        # Reduce number of 3D points to plot
        slicing_percentage = 0.2 #percentage
        slice_points = True

        xp,yp,zp = [],[],[]
        if plot_points:
            for kfID in range(1,self.keyframe_ID+1):
                points = self.measurements[kfID]["points3d"]

                if slice_points:
                    count = int(len(points)*slicing_percentage)
                    indices = np.random.choice(range(0,len(points)),
                                                count, replace=False)
                    for idx in indices:
                        xp.append(points[idx][0])
                        yp.append(points[idx][1])
                        zp.append(points[idx][2])
                else:
                    for pt in points:
                        xp.append(pt[0])
                        yp.append(pt[1])
                        zp.append(pt[2])

        if plot3D:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot3D(xc, zc, yc, 'green')
            #s1 = [2*2**n for n in range(len(xc))]
            ax.scatter(xc, zc, yc, c='red')
            if plot_points:
                s2 = [2*1**n for n in range(len(xp))]
                ax.scatter(xp, zp, yp, c='gray',s=s2)
            ax.set_zlim([-10,10])
        else:
            s1 = [2*2**n for n in range(len(xc))]
            plt.plot(xc,zc,'green')
            plt.scatter(xc,zc,c='red')
            if plot_points:
                s2 = [2*1**n for n in range(len(xp))]
                plt.scatter(xp,zp,c='gray',s=s2)
        plt.show()


    def write2file(self, sframe, end_frame, path, dataset):
        import json

        print("Writing data to ~/{}".format(path))

        for f in range(sframe,end_frame):
            pose_file = open(path+"poses_"+dataset+".txt","a+")
            pose_json = json.dumps(self.poses[f])
            pose_file.write(pose_json + "\n")

            point_file = open(path+"point_data_"+dataset+".txt","a+")
            point_json = json.dumps(self.point_data[f])
            point_file.write(point_json + "\n")
        print("Files wrtten\n")



if __name__ == '__main__':
    # Dataset
    dataset = "kitti"
    
    imgL_files = sorted(glob("data/"+dataset.lower()+"/image_0/*.png"))
    imgR_files = sorted(glob("data/"+dataset.lower()+"/image_1/*.png"))

    if dataset.lower() == "kitti":
        # format: <fx fy cx cy width height baseline>
        intrinsics = [718.856,718.856,607.1928,185.2157,1241,376,0.5371657]
        dist_coeffs = None # Or np.zeros((4,1))
    else:
        # Specify intrinsics of other camera/dataset used
        print("Intrinsics required"); sys.exit()

    show_trajectory = False
    show_3Dpts = True
    plot3D = False
    write2file = False

    prev_imgL, prev_imgR = None,None
    total_duration = 0
    initialized = False

    svo = Stereo_PnPVO(intrinsics, dist_coeffs)

    print()
    start_frame = 0; end_frame = 4 #len(imgL_files)
    for i in range(start_frame,end_frame):
        print("frame_ID->{}".format(i))
        start_time = time.time()

        imgL = cv2.imread(imgL_files[i])
        imgR = cv2.imread(imgR_files[i])

        frame = Frame(i, imgL, imgR, intrinsics, dist_coeffs)
        frame.extract_features()

        if not initialized:
            # Get points
            svo.initialize(frame)
            initialized = True

        else:
            # Track
            svo.track(frame, prev_imgL, prev_imgR)

        # Keep previous frames for tracking
        prev_imgL = imgL.copy()
        prev_imgR = imgR.copy()

        # local and global durations
        duration = time.time()-start_time
        total_duration += duration
        print("duration: {}s".format(duration)); print()

    print("--------------------------------------------------------------")
    print("Total duration: {}s".format(total_duration))
    print("No. of keyframes:", svo.keyframe_ID)
    print("No. of frames:", end_frame - start_frame); print()


    # Write pose and points to file
    if write2file:
        path = "outputs/"
        if os.path.exists(path+"poses_"+dataset+".txt"):
            os.remove(path+"poses_"+dataset+".txt")
        if os.path.exists(path+"point_data_"+dataset+".txt"):
            os.remove(path+"point_data_"+dataset+".txt")
        svo.write2file(start_frame, end_frame, path, dataset)

    # Plot camera trajectory
    if show_trajectory:
        print("Plotting trajectory...\n")
        svo.plot_trajectory(start_frame, end_frame, plot3D, show_3Dpts)

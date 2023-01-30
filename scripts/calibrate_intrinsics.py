import numpy as np
import cv2
import glob
import os
import argparse
import math
from termcolor import cprint
import sys
from tqdm import tqdm

class ChessboardCalibrator(object):

    def __init__(self, w, h, square_size_meter):
        self.w = w
        self.h = h
        self.square_size_meter = square_size_meter
        self.chessboard_points = np.zeros((w * h, 3), np.float32)
        self.chessboard_points[:, :2] = np.mgrid[0:w,0:h].T.reshape(-1, 2) * square_size_meter
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


        self.kernel_size = 11

        self.images = []
        self.gray_images = []
        self.undistored_images = []
        self.chessboard_world_points = []
        self.chessboard_image_points = []
        self.camera_matrix = np.eye(3, dtype=float)
        self.camera_matrix_refined = np.eye(3, dtype=float)
        self.camera_distortions = np.zeros((5,), dtype=float)
        self.chessboard_poses = []
        self.robot_poses = []

    def consumeImage(self, img, color_image=True):
        if color_image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)


        if ret:
            self.images.append(img.copy())
            self.gray_images.append(gray.copy())
            self.chessboard_world_points.append(self.chessboard_points)
            corners2 = cv2.cornerSubPix(gray, corners, (self.kernel_size, self.kernel_size), (-1, -1), self.criteria)
            self.chessboard_image_points.append(corners2)

            return corners2
        return None

    def calibrate(self, undistortion_alpha=1.0):
        if len(self.images) > 0:
            ret, self.camera_matrix, self.camera_distortions, rvecs, tvecs = cv2.calibrateCamera(
                self.chessboard_world_points,
                self.chessboard_image_points,
                self.gray_images[0].shape[::-1],
                None, None
            )

            self.chessboard_poses = []
            for i, t in enumerate(tvecs):
                rvec = rvecs[i]
                R, _ = cv2.Rodrigues(rvec)
                T = np.hstack((R, t))
                T = np.vstack((T, np.array([0, 0, 0, 1.0])))
                self.chessboard_poses.append(T)

            self.refineCameraMatrix(alpha=undistortion_alpha)
            self.buildUndistortedImages()
            return True
        return False

    def refineCameraMatrix(self, alpha=1):
        if len(self.images) > 0:
            img = self.images[0]
            h,  w = img.shape[:2]
            self.camera_matrix_refined, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.camera_distortions,
                (w, h), alpha, (w, h)
            )

    def buildUndistortedImages(self):
        for img in self.images:
            dst = cv2.undistort(img,
                                self.camera_matrix,
                                self.camera_distortions,
                                None,
                                self.camera_matrix_refined
                                )
            self.undistored_images.append(dst)


    def singleRowPrint(self, reference_image, camera_matrix, camera_distortions):
        data = [
            reference_image.shape[1],
            reference_image.shape[0],
            camera_matrix[0, 0],
            camera_matrix[1, 1],
            camera_matrix[0, 2],
            camera_matrix[1, 2]
        ]
        data.extend(camera_distortions.ravel().tolist())
        return np.array(data)

    def setRobotPoses(self, poses):
        if len(poses) != len(self.images):
            print(len(poses))
            print(len(self.images))
            print("Error! Number of Robot Poses is different from the number of images")
            #return
        self.robot_poses = poses


    
    def run_calibration(self, images_files, args):

        #######################################
        # Feed with images
        #######################################
        self.discarded = []
        with tqdm(total=len(images_files)) as pbar:
            for i, fname in enumerate(images_files):
                img = cv2.imread(fname)
                if self.consumeImage(img) is None:
                    self.discarded.append(i)
                pbar.update()

        print("DIscarded images:", len(self.discarded))
        #######################################
        # Images check
        #######################################
        if len(self.images) == 0:
            cprint("No images found!")
            sys.exit(0)

        #######################################
        # Calibrate
        #######################################
        cprint("{}\n Calibration...\n {}".format("=" * 50, "=" * 50), color="yellow")
        self.calibrate(undistortion_alpha=args['undistortion_alpha'])

        #######################################
        # Output options
        #######################################
        np.set_printoptions(precision=6, suppress=True, linewidth=np.inf)

        reference_image = self.images[0]
        first_color = "red"
        separator_width = 50
        cprint("First Calibration {}".format("=" * separator_width), first_color)

        cprint("\nCamera Matrix", first_color)
        print(self.camera_matrix)

        cprint("\nCamera Distortions", first_color)
        print(self.camera_distortions)

        cprint("\nSingle row", first_color)
        print(np.array2string(self.singleRowPrint(reference_image, self.camera_matrix, self.camera_distortions), separator=","))


        first_color = "green"
        cprint("\nRefined Calibration {}".format("=" * separator_width), first_color)

        cprint("\nCamera Matrix", first_color)
        print(self.camera_matrix_refined)

        cprint("\nCamera Distortions", first_color)
        print(self.camera_distortions)

        cprint("\nSingle row", first_color)
        print(np.array2string(self.singleRowPrint(reference_image, self.camera_matrix_refined, self.camera_distortions), separator=","))



    def run_undistort(self, images_files, camera_matrix, camera_distortion):

        self.camera_matrix = camera_matrix
        self.camera_matrix_refined = camera_matrix
        self.camera_distortions = camera_distortion

        #######################################
        # Feed with images
        #######################################
        self.discarded = []
        with tqdm(total=len(images_files)) as pbar:
            for i, fname in enumerate(images_files):
                img = cv2.imread(fname)
                if self.consumeImage(img) is None:
                    self.discarded.append(i)
                pbar.update()

        self.buildUndistortedImages()



if __name__ == "__main__":

    #######################################
    # Arguments
    #######################################
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_folder", required=False, default="manual_saved_frames", help="Image Folder")
    ap.add_argument("--image_extension", default='png', type=str)
    ap.add_argument("--chessboard_size", default='7x5', type=str)
    ap.add_argument("--chessboard_square_size", default=0.025, type=float)
    ap.add_argument("--save_undistorted", action='store_true')
    ap.add_argument("--undistortion_alpha", default=1.0, type=float)
    args = vars(ap.parse_args())


    #######################################
    # Creates Calibrator
    #######################################
    w, h = map(int, args['chessboard_size'].split('x'))
    sqs = args['chessboard_square_size']
    chessboard_calibrator = ChessboardCalibrator(w, h, sqs)

    #######################################
    # Load Images Files
    #######################################
    images_files = sorted(glob.glob(os.path.join(args['image_folder'], '*.' + args['image_extension'])))


    chessboard_calibrator.run_calibration(images_files, args)
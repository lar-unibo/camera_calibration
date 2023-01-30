import numpy as np
import glob, os, argparse, cv2
from termcolor import cprint
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from tqdm import tqdm




######################################################
class ChessboardCalibrator(object):

    def __init__(self, w, h, square_size_meter):
        self.w = w
        self.h = h
        self.square_size_meter = square_size_meter
        self.chessboard_points = np.zeros((w * h, 3), np.float32)
        self.chessboard_points[:, :2] = np.mgrid[0:w,0:h].T.reshape(-1, 2) * square_size_meter
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.kernel_size = 11
        self.camera_matrix = np.eye(3, dtype=float)
        self.camera_matrix_refined = np.eye(3, dtype=float)
        self.camera_distortions = np.zeros((5,), dtype=float)

        self.images = []
        self.undistored_images = []
        self.gray_images = []
        self.chessboard_world_points = []
        self.chessboard_image_points = []

        self.chessboard_poses = []
        self.robot_poses = []


    def validateImage(self, img, robot_pose, color_image=True):

        if color_image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)

        if ret:
            self.images.append(img.copy())
            self.robot_poses.append(robot_pose)
            self.gray_images.append(gray.copy())
            self.chessboard_world_points.append(self.chessboard_points)
            
            corners2 = cv2.cornerSubPix(gray, corners, (self.kernel_size, self.kernel_size), (-1, -1), self.criteria)
            self.chessboard_image_points.append(corners2)

        else:
            print("not valid!")


    def calibrate(self):
        if len(self.images) > 0:
            ret, self.camera_matrix, self.camera_distortions, rvecs, tvecs = cv2.calibrateCamera(
                self.chessboard_world_points,
                self.chessboard_image_points,
                self.gray_images[0].shape[::-1],
                None, None
            )
            print("RMS Error: ", ret)

            self.chessboard_poses = []
            for i, t in enumerate(tvecs):
                rvec = rvecs[i]
                R, _ = cv2.Rodrigues(rvec)
                T = np.hstack((R, t))
                T = np.vstack((T, np.array([0, 0, 0, 1.0])))
                self.chessboard_poses.append(T)

                uimg = self.undistort(self.images[i], self.camera_matrix, self.camera_distortions)
                self.undistored_images.append(uimg)

            return True
        return False


    def undistort(self, img, camera_matrix, camera_dist):
        h,  w = img.shape[:2]
        camera_matrix_refined, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, camera_matrix, camera_dist, None, camera_matrix_refined)
        return dst

    
    def validateUndistortedImages(self):
        self.images = []
        self.chessboard_image_points = []
        self.chessboard_world_points = []
        self.robot_poses_val = []
        for it, img in enumerate(self.undistored_images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (self.w, self.h), None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (self.kernel_size, self.kernel_size), (-1, -1), self.criteria)

                self.images.append(img.copy())
                self.robot_poses_val.append(self.robot_poses[it])
                self.chessboard_world_points.append(self.chessboard_points)              
                self.chessboard_image_points.append(corners2)




######################################################

def fromPoseToMatrix(pose):
    T = np.eye(4)
    T[:3,:3] = Rotation.from_quat([pose[-4], pose[-3], pose[-2], pose[-1]]).as_matrix()
    T[:3,3] = np.array([pose[0], pose[1], pose[2]])
    return T

def optimizationFunction(x, robot_poses, pattern_poses, offset_factor=[0.0, 0.0, 0.0]):
    e = 0.0

    T_find = fromPoseToMatrix(x)

    T_k = np.eye(4)
    T_k[:3,3] = np.array([offset_factor[0], offset_factor[1], offset_factor[2]])

    for i in range(0, len(robot_poses)):
        for j in range(0, len(robot_poses)):
            if i != j:
                T1_i = robot_poses[i]
                T2_i = pattern_poses[i]
                T1_j = robot_poses[j]
                T2_j = pattern_poses[j]
                Tr_i = np.matmul(np.matmul(np.matmul(T1_i,T_find),T2_i),T_k)
                Tr_j = np.matmul(np.matmul(np.matmul(T1_j,T_find),T2_j),T_k)
                pr_i = Tr_i[:3,3]
                pr_j = Tr_j[:3,3]
                diff = pr_j - pr_i
                e = e + np.linalg.norm(diff)**2
    return e




def calibrate_extrinsics(initial_guess, chessboard_calibrator, debug=False):

        chessboard_calibrator.validateUndistortedImages()

        chessboard_poses = []
        robot_poses = []
        for i, chessboard_pose in enumerate(chessboard_calibrator.chessboard_poses):
            chessboard_poses.append(chessboard_pose)
            robot_poses.append(chessboard_calibrator.robot_poses[i])


            if debug:
                print(" ")
                print("i: ", i)
                print("original pose t: ", chessboard_pose[:3,3].ravel())

                origin = tuple(chessboard_calibrator.chessboard_image_points[i][0].ravel().astype(int))

                points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
                rvec, _ = cv2.Rodrigues(chessboard_pose[:3,:3])
                axis_points, _ = cv2.projectPoints(points, rvec, chessboard_pose[:3,3], chessboard_calibrator.camera_matrix, (0, 0, 0, 0))
                axis_points = np.squeeze(axis_points, axis=1)
                dirs = [np.array([a[0] - origin[0], a[1] - origin[1]]) for a in axis_points]
                axis_points = [np.array(origin) + 100 * d/np.linalg.norm(d) for d in dirs]

                plt.imshow(chessboard_calibrator.images[i])
                plt.scatter(origin[0], origin[1])
                plt.plot([origin[0], axis_points[0][0]], [origin[1], axis_points[0][1]], c="red")
                plt.plot([origin[0], axis_points[1][0]], [origin[1], axis_points[1][1]], c="green")
                plt.plot([origin[0], axis_points[2][0]], [origin[1], axis_points[2][1]], c="blue")
                plt.show()


  
        x0 = initial_guess
        translation_bounds = [-0.3, 0.3, -0.3, 0.3, -0.3, 0.3]
        offset_factor = [0.0, 0.0, 0.0]

        bounds = [
            (translation_bounds[0], translation_bounds[1]),
            (translation_bounds[2], translation_bounds[3]),
            (translation_bounds[4], translation_bounds[5]),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf)
        ]

        cprint("{}\n Exstrinsics Computation...\n {}".format("=" * 50, "=" * 50), color="yellow")

        cprint("Initial Guess: {}".format(x0), 'blue')
        cprint("Bounds: {}".format(bounds), 'blue')
        cprint("Offset Factor: {}".format(offset_factor), 'blue')

        degs = Rotation.from_quat([x0[3], x0[4], x0[5], x0[6]]).as_euler('xyz', degrees=True)
        print("initial guess rotation: ", degs)

        res = minimize(optimizationFunction, x0,
                    args=(robot_poses, chessboard_poses, offset_factor),
                    method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000000, 'disp': 0})


        quat_normalized = Rotation.from_quat([res.x[3], res.x[4], res.x[5], res.x[6]]).as_quat()

        first_color = "blue"
        cprint("\nExtrinsics {}".format("=" * 50), first_color)
        cprint("\nCamera Frame: [X,Y,Z,QX,QY,QZ,QW]", first_color)
        print("position: ", [res.x[0], res.x[1], res.x[2]])
        print("quaternions: ", quat_normalized)

        cprint("\nCamera Frame orientation: [Roll Pitch Yaw] deg", first_color)
        degs = Rotation.from_quat([res.x[3], res.x[4], res.x[5], res.x[6]]).as_euler('xyz', degrees=True)
        print(degs)

        cprint("\nCamera Frame orientation: [Roll Pitch Yaw] rad", first_color)
        rads = Rotation.from_quat([res.x[3], res.x[4], res.x[5], res.x[6]]).as_euler('xyz', degrees=False)
        print(rads)




if __name__ == "__main__":

    #######################################
    # Arguments
    #######################################
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_folder", required=False, default="/home/lar/ros/del_ws/src/calibration/manual_saved_frames_1")
    ap.add_argument("--image_extension", default='png', type=str)
    ap.add_argument("--chessboard_size", default='7x6', type=str)
    ap.add_argument("--chessboard_square_size", default=0.017, type=float)
    ap.add_argument("--flag_depth", default=True, type=bool)
    args = vars(ap.parse_args())
    print(args)
    
    # Initial Guess for Extrinsics Optimization
    initial_guess = np.array([0.046839, 0.000012, 0.077146, 0.019076, 0.014965, 0.690177, 0.723234]) 

    ##########################################################################################
    # Load Images Files
    #######################################
    images_files = sorted(glob.glob(os.path.join(args['image_folder'], '*.' + args['image_extension'])))
    poses_files = []
    for f in images_files:
        name = f.split("/")[-1].split(".")[0]
        poses_files.append(os.path.join(args['image_folder'], name + ".txt"))


    #######################################
    # Load
    #######################################
    robot_poses = [np.loadtxt(pose_file) for pose_file in poses_files]
    images = [cv2.imread(f, cv2.IMREAD_COLOR) for f in images_files]



    #######################################
    # Creates Calibrator
    #######################################
    w, h = map(int, args['chessboard_size'].split('x'))
    sqs = args['chessboard_square_size']
    calib = ChessboardCalibrator(w, h, sqs)


    #######################################
    # Filter Images where chessboard not visible
    #######################################
    with tqdm(total=len(images)) as pbar:
        for it, img in enumerate(images):
            pose = robot_poses[it]
            T_robot = fromPoseToMatrix(pose) # ee_link -> world
            calib.validateImage(img, T_robot)
            pbar.update()

    rv = calib.calibrate()
    print("calibrate return value: ", rv)

    np.set_printoptions(precision=6, suppress=True, linewidth=np.inf)

    cprint("First Calibration {}".format("=" * 50), "red")

    cprint("\nCamera Matrix", "red")
    print(calib.camera_matrix)

    cprint("\nCamera Distortions", "red")
    print(calib.camera_distortions)

    cprint("\nRefined Calibration {}".format("=" * 50), "green")

    cprint("\nCamera Matrix", "green")
    print(calib.camera_matrix_refined)

    cprint("\nCamera Distortions", "green")
    print(calib.camera_distortions)


    #######################################
    # Extrinsics
    #######################################
    calibrate_extrinsics(initial_guess, calib)
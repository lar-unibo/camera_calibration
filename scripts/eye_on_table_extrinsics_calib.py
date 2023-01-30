import numpy as np
import glob, os, math, sys, argparse, cv2
from termcolor import cprint
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

from scripts.calibrate_intrinsics import ChessboardCalibrator


def fromPoseToMatrix(pose):
    T = np.eye(4)
    T[:3,:3] = Rotation.from_quat([pose[-4], pose[-3], pose[-2], pose[-1]]).as_matrix()
    T[:3,3] = np.array([pose[0], pose[1], pose[2]])
    return T

def quatFromMatrix(T):
    return Rotation.from_matrix(T[:3,:3]).as_quat()


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



def bundleAdjustment(x, mean_chessboard_pose, chessboard_calibrator_undistorted, debug=True):

    T_find = x
    e = 0.0
    for index in range(0, len(chessboard_calibrator_undistorted.images)):
        img = chessboard_calibrator_undistorted.images[index]

        img_pts = chessboard_calibrator_undistorted.chessboard_image_points[index]

        num = chessboard_calibrator_undistorted.w * chessboard_calibrator_undistorted.h

        camera_pose = np.matmul(chessboard_calibrator_undistorted.robot_poses[index], T_find)
        #camera_pose_inv = np.linalg.inv(camera_pose)

        if debug:
            print("IMAGE_POINTS")
            img1 = img.copy()
            cv2.drawChessboardCorners(img1, (chessboard_calibrator_undistorted.w,
                                            chessboard_calibrator_undistorted.h), img_pts, True)
            cv2.namedWindow("img1", 0)
            cv2.imshow("img1", img1)


        Tvec = np.array(camera_pose[0:3, 3])
        Rvec, _ = cv2.Rodrigues(camera_pose[:3,:3]) 

        world_points = chessboard_calibrator_undistorted.chessboard_world_points[index]

        world_points = np.hstack((world_points, np.ones((num, 1))))

        world_points = np.matmul(mean_chessboard_pose, world_points.T).astype(np.float32)
        world_points = world_points[:3, :].T

        proj_points, _ = cv2.projectPoints(world_points, Rvec, Tvec, chessboard_calibrator_undistorted.camera_matrix,
                                           chessboard_calibrator_undistorted.camera_distortions)

        proj_points = np.array(proj_points)
        diff = np.linalg.norm(np.square(img_pts - proj_points))

        if debug:
            img2 = img.copy()
            cv2.drawChessboardCorners(img2, (chessboard_calibrator_undistorted.w,
                                            chessboard_calibrator_undistorted.h), proj_points, True)
            print("DIFFERENCE", diff)
            cv2.namedWindow("img2", 0)
            cv2.imshow("img2", img2)
            cv2.waitKey(0)


        e += diff

    return e



def calibrate_extrinsics(initial_guess, robot_poses, chessboard_calibrator, w, h, sqs, depth_files, camera_matrix):

        #######################################
        # Creates Undistorted Chessboard Calibrator
        #######################################
        chessboard_calibrator_undistorted = ChessboardCalibrator(w, h, sqs)
        robot_poses_undistorted = []
        for it, uimg in enumerate(chessboard_calibrator.images):
            rv = chessboard_calibrator_undistorted.consumeImage(uimg, depth=None)
            if rv is not None:
                robot_poses_undistorted.append(robot_poses[it])
                

        cprint("{}\n Compute Chessboard Poses...\n {}".format("=" * 50, "=" * 50), color="yellow")
        chessboard_calibrator_undistorted.calibrate()

        #######################################
        # Set Robot Poses
        #######################################
        chessboard_calibrator_undistorted.setRobotPoses(robot_poses_undistorted)

        initial_guess = np.array(initial_guess)  
        T_init_guess = fromPoseToMatrix(initial_guess)
        print("initial guess: ", initial_guess)

        chessboard_poses = []
        camera_poses = []
        for i, robot_pose in enumerate(chessboard_calibrator_undistorted.robot_poses):
            camera_pose = np.matmul(robot_pose, T_init_guess)
            camera_poses.append(camera_pose)
            print("camera_pose: ", camera_pose)

            chessboard_pose = chessboard_calibrator_undistorted.chessboard_poses[i]
            print("original pose t: ", chessboard_pose[:3,3].ravel())

            chessboard_pose = np.matmul(camera_pose, chessboard_pose)
            chessboard_poses.append(chessboard_pose)
            print("chessboard_poses: ", chessboard_pose)

  
        x0 = initial_guess
        translation_bounds = [0.5, 1, 0.0, 0.3, 0.5, 1.0]
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
                    args=(chessboard_calibrator_undistorted.robot_poses,
                            chessboard_calibrator_undistorted.chessboard_poses,
                            offset_factor),
                    method='L-BFGS-B', bounds=bounds, options={'maxiter': args['max_iterations'], 'disp': args['optimization_debug']})


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
    ap.add_argument("--image_folder", required=False, default="/home/lar/ros/del_ws/src/calibration/manual_saved_frames_10")
    ap.add_argument("--image_extension", default='png', type=str)
    ap.add_argument("--depth_extension", default='exr', type=str)
    ap.add_argument("--chessboard_size", default='7x6', type=str)
    ap.add_argument("--chessboard_square_size", default=0.017, type=float)
    ap.add_argument("--save_undistorted", action='store_true')
    ap.add_argument("--compute_extrinsics", action='store_true', default=True)
    ap.add_argument("--undistortion_alpha", default=1.0, type=float)
    ap.add_argument('--max_iterations', default=10000000, type=int)
    ap.add_argument('--optimization_debug', default=0, type=int)

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
    depth_files = sorted(glob.glob(os.path.join(args['image_folder'], '*.' + args['depth_extension'])))


    #######################################
    # Load Robot Poses
    #######################################
    poses_files = sorted(glob.glob(os.path.join(args['image_folder'], '*.txt')))
    robot_poses = [np.loadtxt(pose_file) for pose_file in poses_files]


    #######################################
    # Undistort images
    #######################################
    K = np.array([[1723.86, 0, 823.559], [0, 1723.83, 598.321], [0, 0, 1]])
    D = np.array([-0.157741, 0.170455, 0.000386184, -0.000158724, -0.0760488])
    W = 1680
    H = 1200

    chessboard_calibrator.run_undistort(images_files, camera_matrix=K, camera_distortion=D)

    # remove discarded poses
    robot_poses_filt = []
    depth_files_filt = []
    for i, p in enumerate(robot_poses):
        if i not in chessboard_calibrator.discarded:
            
            depth_files_filt.append(depth_files[i])

            T_robot = fromPoseToMatrix(p)
            #T_robot = np.linalg.inv(T_robot)
            robot_poses_filt.append(T_robot)


    #######################################
    # Extrinsics
    #######################################
    initial_guess = np.array([0.728107, 0.147628, 0.709791, 0.738343, 0.67053, -0.051192, -0.071317]) 
    calibrate_extrinsics(initial_guess, robot_poses_filt, chessboard_calibrator, w, h, sqs, depth_files_filt, camera_matrix=K)
import numpy as np
import cv2

def find_chessboard_points(
    frame,
    pattern_size=(8, 6),
    corners=None,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
):
    """Finds chessboard on a picture and refines results.

    Args:
        frame: picture, video frame or sth.
        patterm_size: number of squares in a chessboard.
        corners: i think its useless. idk
        criteria: criteria for refining.
    
    Returns:
        Points of the chessboard squares on a picture
    """
    ret, corners = cv2.findChessboardCorners(frame, pattern_size, corners)
    if not ret:
        return None

    corners = cv2.cornerSubPix(frame, corners, (11, 11), (-1, -1), criteria)
    return corners


def get_intrinsic_params(
    object_points,
    image_points,
    image_shape,
    camera_matrix=None,
    distortion_matrix=None,
    **kwargs
):
    """Generating intrinsic parameters of camera.

    Args:
        object_points: points in 3d space.
        image_points: points on a 2d picture that coresponds to 3d points
        image_shape: (width, height) of a picture
        camera_matrix: i think its useless. idk
        distortion_matrix: i think its useless. idk

    Returns:
        mtx: camera matrix.
        distortion: camera distortion matrix.
    """
    ret, mtx, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_shape,
        camera_matrix,
        distortion_matrix,
        **kwargs
    )
    if not ret:
        return None

    return mtx, distortion

def get_extrinsic_params(
    object_points,
    image_points,
    camera_matrix=None,
    distortion_matrix=None,
):
    """Generating extrinsic parameters of camera in 3d.
    
    Args:
        object_points: points in 3d space.
        image_points: points on a 2d picture that coresponds to 3d points
        image_shape: (width, height) of a picture
        camera_matrix: matrix of a camera :)
        distortion_matrix: distortion matrix of a camera :)

    Returns:
        rotation matrix of a camera in 3d space.
        position vector of a camera in 3d space.
    """
    ret, rotation_vecs, translation_vecs = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion_matrix
    )

    if not ret:
        return None

    return cv2.Rodrigues(rotation_vecs)[0], translation_vecs


def get_3d_point_from_stereo_cameras(
    intrinsic_params_mtx_1,
    camera_rotation_1,
    camera_translation_1,
    intrinsic_params_mtx_2,
    camera_rotation_2,
    camera_translation_2,
    image1_point_loc,
    image2_point_loc,
):
    """Calculating point in 3d space using point in 2 2d images.

    Args:
        intrinsic_params_mtx_1: first camera matrix.
        camera_rotation_1: first camera rotation matrix in 3d space.
        camera_translation_1: first camera position in 3d space.
        intrinsic_params_mtx_2: second camera matrix.
        camera_rotation_2: second camera rotation matrix in 3d space.
        camera_translation_2: second camera position in 3d space.
        image1_point_loc: 2d point on a first image.
        image2_point_loc: 2d point on a second image.

    Returns:
        position of a point in 3d space.
    """
    pnt1 = np.array([[image1_point_loc[0]], [image1_point_loc[1]], [1]])
    pnt2 = np.array([[image2_point_loc[0]], [image2_point_loc[1]], [1]])

    point1p = intrinsic_params_mtx_1 @ pnt1
    point2p = intrinsic_params_mtx_2 @ pnt2

    A = np.array(
        [
            [
                camera_rotation_1[2][0] * point1p[0][0] - camera_rotation_1[0][0],
                camera_rotation_1[2][1] * point1p[0][0] - camera_rotation_1[0][1],
                camera_rotation_1[2][2] * point1p[0][0] - camera_rotation_1[0][2],
            ],
            [
                camera_rotation_1[2][0] * point1p[1][0] - camera_rotation_1[1][0],
                camera_rotation_1[2][1] * point1p[1][0] - camera_rotation_1[1][1],
                camera_rotation_1[2][2] * point1p[1][0] - camera_rotation_1[1][2],
            ],
            [
                camera_rotation_2[2][0] * point2p[0][0] - camera_rotation_2[0][0],
                camera_rotation_2[2][1] * point2p[0][0] - camera_rotation_2[0][1],
                camera_rotation_2[2][2] * point2p[0][0] - camera_rotation_2[0][2],
            ],
            [
                camera_rotation_2[2][0] * point2p[1][0] - camera_rotation_2[1][0],
                camera_rotation_2[2][1] * point2p[1][0] - camera_rotation_2[1][1],
                camera_rotation_2[2][2] * point2p[1][0] - camera_rotation_2[1][2],
            ],
        ]
    )
    B = np.array(
        [
            [camera_translation_1[0][0] - camera_translation_1[2][0] * point1p[0][0]],
            [camera_translation_1[1][0] - camera_translation_1[2][0] * point1p[1][0]],
            [camera_translation_2[0][0] - camera_translation_2[2][0] * point2p[0][0]],
            [camera_translation_2[1][0] - camera_translation_2[2][0] * point2p[1][0]],
        ]
    )

    return np.linalg.lstsq(A, B, rcond=None)[0].T[0]  # 3d position


def get_3d_point_from_n_cameras(
    intrinsic_params_mtxs,
    camera_rotations,
    camera_translations,
    image_point_locs,
):
    """Calculating point in 3d space using point in n 2d images.

    Args:
        intrinsic_params_mtxs: list of camera matrices.
        camera_rotations: list of camera rotations matrices in 3d space.
        camera_translations: list of camera positions in 3d space.
        image_point_locs: list of 2d positions on a pictures.

    Returns:
        position of a point in 3d space.
    """
    assert len(intrinsic_params_mtxs) == len(camera_rotations) == len(camera_translations) == len(image_point_locs), "all lists must have the same lenght"
    assert len(intrinsic_params_mtxs) >= 2, "required at least 2 different views (difference not checked)"

    bigA = np.empty((len(intrinsic_params_mtxs)*2,3))
    bigB = np.empty((len(intrinsic_params_mtxs)*2,1))

    for i in range(len(intrinsic_params_mtxs)):

        image_point_loc_multiplied = intrinsic_params_mtxs[i] @ np.array([[image_point_locs[i][0]], [image_point_locs[i][1]], [1]])

        bigA[i * 2:i * 2 + 2, :] = np.array(
            [
                [
                    camera_rotations[i][2][0] * image_point_loc_multiplied[0][0] - camera_rotations[i][0][0],
                    camera_rotations[i][2][1] * image_point_loc_multiplied[0][0] - camera_rotations[i][0][1],
                    camera_rotations[i][2][2] * image_point_loc_multiplied[0][0] - camera_rotations[i][0][2],
                ],
                [
                    camera_rotations[i][2][0] * image_point_loc_multiplied[1][0] - camera_rotations[i][1][0],
                    camera_rotations[i][2][1] * image_point_loc_multiplied[1][0] - camera_rotations[i][1][1],
                    camera_rotations[i][2][2] * image_point_loc_multiplied[1][0] - camera_rotations[i][1][2],
                ]
            ]
        )

        bigB[i * 2:i * 2 + 2, :] = np.array(
            [
                [camera_translations[i][0][0] - camera_translations[i][2][0] * image_point_loc_multiplied[0][0]],
                [camera_translations[i][1][0] - camera_translations[i][2][0] * image_point_loc_multiplied[1][0]]
            ]
        )

    return np.linalg.lstsq(bigA, bigB, rcond=None)[0].T[0]  # 3d position

class UndistortionAlgorithm:

    FAST = 0
    SLOW = 1

    def __init__(self, camera_matrix, camera_distortion, size):# size (w, h)
        """Initializing undistorion algorithm variables.

        Args:
            camera_matrix: matrix of a camera :)
            camera_distortion: distortion matrix of a camera :)
            size: size of a frame (width, height)
        """
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion

        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, camera_distortion, size, 1, size)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.camera_matrix, self.camera_distortion, None, self.newcameramtx, size, 5)

    def undistort_frame(self, frame, speed=0):
        """Undistorts given image.
        
        Args:
            frame: picture, video frame or sth.
            speed: quality of undistortion slower-better :)

        Returns:
            Undistorted frame.
        """
        assert speed == self.FAST or speed == self.SLOW, "invalid speed"
        match speed:
            case self.FAST:
                return cv2.undistort(frame, self.camera_matrix, self.camera_distortion, None, self.newcameramtx)
            case self.SLOW:
                return cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
            case _:
                return None
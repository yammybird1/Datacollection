import pyzed.sl as sl
import math
import numpy as np
import cv2
import os
import sys
import tifffile as tf
import datetime
import open3d as o3d
import pptk

# Option 1 is for image capture, depth, confidence and point cloud
# Option 2 is for video recording
# Option 3 is for playback


exposure_steps = [0.5, 0.75, 1.25, 1.5]
num_images_to_collect = len(exposure_steps) +1 # first image is auto
option = 3

# /home/emma/evaluation-of-depth-cameras/apple-data
# dirname = 'apple-data/{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
dirname = '{}'.format(sys.argv[1].split('.')[0])
# dirname = str(timestamp.get_milliseconds())
os.makedirs(dirname, exist_ok=True)
# create a folder inside folder with timestamp inside apple-data
os.makedirs(os.path.join(dirname, 'left-images'), exist_ok=True)
# os.makedirs(os.path.join(dirname, 'right-images'), exist_ok=True)
# os.makedirs(os.path.join(dirname, 'confidence-images'), exist_ok=True)
# os.makedirs(os.path.join(dirname, 'pointcloud-images'), exist_ok=True)
# os.makedirs(os.path.join(dirname, 'depth-images'), exist_ok=True)
# os.makedirs(os.path.join(dirname, 'depth-tiff'), exist_ok=True)


def print_camera_information(cam):
    while True:
        res = input("Do you want to display camera information? [y/n]: ")
        if res == "y":
            print()
            print("Distorsion factor of the right cam before calibration: {0}.".format(
                cam.get_camera_information().calibration_parameters_raw.right_cam.disto))
            print("Distorsion factor of the right cam after calibration: {0}.\n".format(
                cam.get_camera_information().calibration_parameters.right_cam.disto))

            print("Confidence threshold: {0}".format(cam.get_runtime_parameters().confidence_threshold))
            print("Depth min and max range values: {0}, {1}".format(cam.get_init_parameters().depth_minimum_distance,
                                                                    cam.get_init_parameters().depth_maximum_distance)
)
            print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height))
            print("Camera FPS: {0}".format(cam.get_camera_information().camera_fps))
            print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))
            break
        elif res == "n":
            print("Camera information not displayed.\n")
            break
        else:
            print("Error, please enter [y/n].\n")


def saving_image(key, mat, i):
    img = sl.ERROR_CODE.FAILURE
    while img != sl.ERROR_CODE.SUCCESS:
        filepath = os.path.join(dirname, 'frame_{:02d}.jpg'.format(i))
        img = mat.write(filepath)
        print("Saving image : {0}".format(repr(img)))
        if img == sl.ERROR_CODE.SUCCESS:
            break
        else:
            print("Help: you must enter the filepath + filename + PNG extension.")


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    
    ######## Camera
    init_params.camera_resolution = sl.RESOLUTION.HD720
    #init_params.camera_resolution = sl.RESOLUTION.HD2K
    #init_params.camera_resolution = sl.RESOLUTION.HD1080
    #init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 30  # Set fps at 30 - other frame rates can be 15, 30, 60, 100
    # Set exposure to a certain % of camera framerate
    #zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
    # Set white balance to a certain K
    #zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITE_BALANCE, 4600)
    # Reset to auto exposure
    #zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)

    ########  Depth
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode (not necessary as this is default), there are three different modes PERFORMANCE, MEDIUM, QUALITY
    #init_params.depth_mode = sl.DEPTH_MODE.QUALITY 
    #init_params.depth_mode = sl.DEPTH_MODE.MEDIUM 
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 1.0
    init_params.depth_maximum_distance = 10.0

    # Open the camera
    if (option == 1 or option == 2):
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit(1)

    if (option == 1):
         
        # Create and set RuntimeParameters after opening the camera
        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
        # Setting the depth confidence parameters
        runtime_parameters.confidence_threshold = 100
        runtime_parameters.textureness_confidence_threshold = 100

        # Capture 150 images and depth, then stop
        i = 0
        image_left = sl.Mat() # stores the images
        image_right = sl.Mat() # stores the images
        depth = sl.Mat() # stores the depth map
        point_cloud = sl.Mat()
        confidence_map = sl.Mat()

        mirror_ref = sl.Transform()
        mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
        tr_np = mirror_ref.m


        while i < num_images_to_collect:
            # A new image is available if grab() returns SUCCESS. This transfers GPU -> CPU by default here
            if i ==0:
                exposure = zed.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)
            else:
                newExposure = exposure*exposure_steps[i-1]
                zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, newExposure)


            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve left image
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                # Retrieve right image
                zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                # Retrieve depth map. Depth is aligned on the overlapped image
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # Retrieve colored point cloud. Point cloud is aligned on the overlapped image.
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                # Retrieve confidence map
                zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)

                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
                print("Image resolution: {0} x {1} || Image timestamp: {2}".format(image_left.get_width(), image_left.get_height(),
                    timestamp.get_milliseconds()))
                print("Image resolution: {0} x {1} || Image timestamp: {2}".format(image_right.get_width(), image_right.get_height(),
                    timestamp.get_milliseconds()))



                # depth_for_display = sl.Mat()
                # zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
                # depth_for_display.write('depth_view.jpg')             

                depth_data = depth.get_data()          
                depth_data[~ np.isfinite(depth_data)] = 0
                depth_img_name = os.path.join(dirname, 'depth_{:02d}.jpg'.format(i))
                cv2.imwrite(depth_img_name, (depth_data/depth_data.max())*255)
                tf.imwrite(depth_img_name.replace('.jpg','.tiff').replace('depth','depth1'), depth_data)

                confidence_data = confidence_map.get_data()
                confidence_img_name = os.path.join(dirname, 'confidence_{:02d}.jpg'.format(i))
                scale = 1./np.max(confidence_data)*255
                cv2.imwrite(confidence_img_name, confidence_data*scale)

                pointcloud_data = point_cloud.get_data()
                pointcloud_name = os.path.join(dirname,'pointcloud_{:02d}.ply'.format(i))
                if not point_cloud.write(pointcloud_name) == sl.ERROR_CODE.SUCCESS:
                    print('point cloud data not saved')

                img_left = image_left.get_data()
                img_name = os.path.join(dirname, 'left_{:02d}.jpg'.format(i))
                cv2.imwrite(img_name, img_left)

                img_right = image_right.get_data()
                img_name = os.path.join(dirname, 'right_{:02d}.jpg'.format(i))
                cv2.imwrite(img_name, img_right)

                np.savez(os.path.join(dirname, 'raw_{:02d}.npz'.format(i)), depth=depth_data, right=img_right, left=img_left, confidence=confidence_data)


                camera_exposure = zed.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)
                camera_whitebalance = zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE)
                print("Exposure time: {:03d}      Whitebalance Temperature: {}".format(camera_exposure, camera_whitebalance))
                i = i + 1

                # for arr in [depth,image_left, image_right, point_cloud, confidence_map ]:
                #     print(arr.get_memory_type())

    elif(option == 2):
        # cd to folder that this script is in
        # type python "datacollection.py" {filename.svo} in terminal
        # go Ctrl-C to stop recording

        if not sys.argv or len(sys.argv) != 2:
            print("Only the path of the output SVO file should be passed as argument.")
            exit(1)

        init_params.depth_mode = sl.DEPTH_MODE.NONE

        path_output = sys.argv[1]
        recording_param = sl.RecordingParameters(path_output, sl.SVO_COMPRESSION_MODE.H264)
        err = zed.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit(1)

        runtime = sl.RuntimeParameters()
        print("SVO is Recording, use Ctrl-C to stop.")
        frames_recorded = 0

        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS :
                frames_recorded += 1
                print("Frame count: " + str(frames_recorded), end="\r")

    elif(option == 3):
        if len(sys.argv) != 2:
            print("Please specify path to .svo file.")
            exit()

        filepath = sys.argv[1]
        print("Reading SVO file: {0}".format(filepath))

        input_type = sl.InputType()
        input_type.set_from_svo_file(filepath)
        init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

        runtime = sl.RuntimeParameters()
        image_left = sl.Mat() # stores the images
        image_right = sl.Mat() # stores the images
        depth = sl.Mat() # stores the depth map
        point_cloud = sl.Mat()
        confidence_map = sl.Mat()
        mat = sl.Mat()

        key = ''
        print("  Quit the video reading:     q\n")
        img_counter = 0
        while key != 113:  # for 'q' key
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                if err == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(mat)
                    cv2.imshow("ZED", mat.get_data())
                    key = cv2.waitKey(1)
                else:
                    key = cv2.waitKey(1)
                # Retrieve left image
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                # Retrieve right image
                
                # zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                # # Retrieve depth map. Depth is aligned on the overlapped image
                # zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # # Retrieve colored point cloud. Point cloud is aligned on the overlapped image.
                #pcd = o3d.geometry.PointCloud()
                # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                # pc = point_cloud.get_data()
                #pcd.points = o3d.utility.Vector3dVector(pc[500:600, 500:600, :3].reshape(-1, 3))
                # pcd = pc[:, :, :3].reshape(-1, 3)
                # v = pptk.viewer(pcd)
                # v.attributes(pcd)
                # v.set(point_size=0.01)

                # # Retrieve confidence map
                # zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)

                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
                print("Image resolution: {0} x {1} || Image timestamp: {2}".format(image_left.get_width(), image_left.get_height(),
                timestamp.get_milliseconds()))
                #print(len(pc))
                # print("Image resolution: {0} x {1} || Image timestamp: {2}".format(image_right.get_width(), image_right.get_height(),
                #     timestamp.get_milliseconds()))

                # depth_for_display = sl.Mat()
                # zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
                # depth_for_display.write('depth_view.jpg')             

                # depth_data = depth.get_data()          
                # depth_data[~ np.isfinite(depth_data)] = 0
                # depth_img_name = os.path.join(dirname, 'depth-images', 'depth_{:02d}.jpg'.format(img_counter))
                # cv2.imwrite(depth_img_name, (depth_data/depth_data.max())*255)
                # depth_tiff_name = os.path.join(dirname, 'depth-tiff', 'depth_{:02d}.tiff'.format(img_counter))
                # tf.imwrite(depth_tiff_name, depth_data)

                # confidence_data = confidence_map.get_data()
                # confidence_img_name = os.path.join(dirname, 'confidence-images','confidence_{:02d}.jpg'.format(img_counter))
                # scale = 1./np.max(confidence_data)*255
                # cv2.imwrite(confidence_img_name, confidence_data*scale)

                # pointcloud_data = point_cloud.get_data()
                # pointcloud_name = os.path.join(dirname, 'pointcloud-images', 'pointcloud_{:02d}.ply'.format(img_counter))
                # if not point_cloud.write(pointcloud_name) == sl.ERROR_CODE.SUCCESS:
                #     print('point cloud data not saved')

                #saving point-cloud
                # pcd = o3d.geometry.PointCloud()
                # output_array = pointcloud_data[:3, :, :]
                # # reshape numpy array from 4D to 3D
                # # Gives a new shape to an array without changing its data
                # pointcloud_data = pointcloud_data[~np.isnan(pointcloud_data)]
                # output_array = output_array.reshape(-1, 3)
                # #output_array = pointcloud_data[:3, :, :].reshape(-1, 3)
                # #output_array = pointcloud_data.reshape(-1, 3)
                # #convert numpy array to open3d
                # pcd.points = o3d.utility.Vector3dVector(output_array)
                # o3d.io.write_point_cloud(pointcloud_name, pcd)
                # o3d.visualization.draw_geometries([pcd])


                img_left = image_left.get_data()
                img_name = os.path.join(dirname, 'left-images', 'left_{:02d}.jpg'.format(img_counter))
                cv2.imwrite(img_name, img_left)

                # img_right = image_right.get_data()
                # img_name = os.path.join(dirname, 'right-images', 'right_{:02d}.jpg'.format(img_counter))
                # cv2.imwrite(img_name, img_right)
                
                img_counter += 1

            else:
                key = cv2.waitKey(1)
        cv2.destroyAllWindows()

        print_camera_information(zed)
        zed.close()
        print("\nFINISH")

    # Close the camera
    if(option == "1"):
        zed.close()

if __name__ == "__main__":
    main()
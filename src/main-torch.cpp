/**
 * @file main.cpp
 * @brief Object Detection and Depth Estimation Server using PyTorch’s C++ API (LibTorch)
 *
 * This file implements an HTTP server that provides object detection (using YOLOv5)
 * and depth estimation (using Depth Anything) capabilities. Both models are loaded as
 * TorchScript modules and run on CPU. The two inference tasks run concurrently using
 * multithreading.
 */

#include <torch/script.h> // LibTorch header for TorchScript
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "httplib.h"
#include "json.hpp"
#include "mjpeg_streamer.hpp"
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cmath>

using json = nlohmann::json;
using MJPEGStreamer = nadjieb::MJPEGStreamer;

// Model file paths (assumed to be TorchScript modules)
#define YOLO_MODEL_PATH "../models/reefscape.torchscript"
#define DEPTH_MODEL_PATH "../models/depth_anything.pt"
#define DEFAULT_CAMERA_DEVICE 0

cv::Mat captured_image;
std::mutex mtx;
cv::Mat frame;
cv::Mat frame_climber_arm;

/**
 * @struct DetectedObject
 * @brief Represents a detected object with spatial information.
 */
struct DetectedObject
{
    std::string objectClass;
    float objectDistance;
    float objectLateralAngle;
    float objectVerticalAngle;
    cv::Rect bbox;
    float confidence;
};

/**
 * @struct ModelResults
 * @brief Container for combined detection and depth estimation results.
 */
struct ModelResults
{
    std::vector<DetectedObject> objects;
};

// YOLO classes (modify according to your model)
// const std::vector<std::string> YOLO_CLASSES = {
//     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//     "hair drier", "toothbrush"};
const std::vector<std::string> YOLO_CLASSES = {
    "Algae", "Coral", "Reef"
};

/**
 * @brief Captures a frame from the specified camera device.
 * @param camera_id Camera device ID (typically 0 for default camera)
 * @return Captured frame as cv::Mat, or an empty Mat on failure.
 */
cv::Mat takeSnapshotFromSelectedCamera(const int camera_id)
{
    // In this example the global "frame" is updated in the video streaming loop.
    return frame;
}

/**
 * @brief Preprocesses image for model input.
 * @param frame Original captured frame.
 * @param target_width Target width for resizing.
 * @param target_height Target height for resizing.
 * @param toFloat Whether to convert the image to floating point (normalized) [default: true].
 * @return Preprocessed image in RGB format.
 */
cv::Mat loadAndPreprocessImage(const cv::Mat &frame, int target_width, int target_height, bool toFloat = true)
{
    cv::Mat img;
    cv::resize(frame, img, cv::Size(target_width, target_height));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    if (toFloat)
    {
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    }
    captured_image = frame.clone();
    return img;
}

/**
 * @brief Runs YOLOv5 object detection on the provided frame using LibTorch.
 * @param[out] class_ids Vector to store detected class IDs.
 * @param[out] confidences Vector to store detection confidence values.
 * @param frame Input frame to process.
 * @return Vector of detected bounding boxes.
 */
std::vector<cv::Rect> runYolo(std::vector<int> &class_ids, std::vector<float> &confidences, const cv::Mat &frame)
{
    std::vector<cv::Rect> bboxes;
    try
    {
        // Load TorchScript model for YOLO
        torch::jit::script::Module model = torch::jit::load(YOLO_MODEL_PATH);
        model.to(torch::kCPU);
        model.eval();

        // Assume YOLO model expects input of size 640x640.
        int target_width = 640;
        int target_height = 640;
        cv::Mat input_img = loadAndPreprocessImage(frame, target_width, target_height, true);

        // Convert cv::Mat (HWC) to a tensor (NCHW)
        auto tensor = torch::from_blob(input_img.data, {1, input_img.rows, input_img.cols, 3});
        tensor = tensor.permute({0, 3, 1, 2});
        tensor = tensor.contiguous();

        // Run inference (no gradients needed)
        torch::NoGradGuard no_grad;
        auto outputs = model.forward({tensor}).toTuple();
        // Assume the first output tensor has shape: [1, num_boxes, box_info_size]
        auto output_tensor = outputs->elements()[0].toTensor();
        auto output_accessor = output_tensor.accessor<float, 3>(); // shape: [1, num_boxes, box_info_size]

        int num_boxes = output_tensor.size(1);
        int box_info_size = output_tensor.size(2);
        float conf_threshold = 0.7f;
        float scale_x = float(captured_image.cols) / float(target_width);
        float scale_y = float(captured_image.rows) / float(target_height);

        for (int i = 0; i < num_boxes; i++)
        {
            float box_conf = output_accessor[0][i][4];
            if (box_conf > conf_threshold)
            {
                int max_class_id = 0;
                float max_class_prob = 0;
                for (int j = 5; j < box_info_size; j++)
                {
                    float prob = output_accessor[0][i][j];
                    if (prob > max_class_prob)
                    {
                        max_class_prob = prob;
                        max_class_id = j - 5;
                    }
                }
                float x = output_accessor[0][i][0] * scale_x;
                float y = output_accessor[0][i][1] * scale_y;
                float w = output_accessor[0][i][2] * scale_x;
                float h = output_accessor[0][i][3] * scale_y;
                int x1 = std::max(0, int(x - w / 2));
                int y1 = std::max(0, int(y - h / 2));
                bboxes.push_back(cv::Rect(x1, y1, int(w), int(h)));
                class_ids.push_back(max_class_id);
                confidences.push_back(box_conf);
            }
        }
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error running YOLO model: " << e.what() << "\n";
    }
    return bboxes;
}

/**
 * @brief Runs Depth Anything model for depth estimation on the provided frame using LibTorch.
 * @param frame Input frame to process.
 * @return 2D vector representing depth map (row-major order).
 */
std::vector<std::vector<float>> runDepthAnything(const cv::Mat &frame)
{
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::vector<float>> depth_map;
    try
    {
        // Load TorchScript model for Depth Anything
        torch::jit::script::Module model = torch::jit::load(DEPTH_MODEL_PATH);
        model.to(torch::kCPU);
        model.eval();

        // Assume Depth model expects input of size 518x518.
        int target_width = 518;
        int target_height = 518;
        cv::Mat input_img = loadAndPreprocessImage(frame, target_width, target_height, true);

        // Convert image to tensor (NCHW)
        auto tensor = torch::from_blob(input_img.data, {1, input_img.rows, input_img.cols, 3});
        tensor = tensor.permute({0, 3, 1, 2});
        tensor = tensor.contiguous();

        torch::NoGradGuard no_grad;
        auto output = model.forward({tensor}).toTensor();
        // Assume output tensor shape is [1, H, W] (or [1,1,H,W])
        int out_height = output.size(1);
        int out_width = output.size(2);
        auto output_accessor = output.accessor<float, 3>(); // shape: [1, H, W]

        depth_map.resize(out_height);
        for (int i = 0; i < out_height; i++)
        {
            depth_map[i].resize(out_width);
            for (int j = 0; j < out_width; j++)
            {
                depth_map[i][j] = output_accessor[0][i][j];
            }
        }
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error running Depth Anything model: " << e.what() << "\n";
    }
    return depth_map;
}

/**
 * @brief Calculates object angles relative to the center of the frame.
 */
void calculateAngles(const cv::Rect &bbox, const cv::Mat &orig_img, float &lateral_angle, float &vertical_angle)
{
    float center_x = bbox.x + bbox.width / 2.0f;
    float center_y = bbox.y + bbox.height / 2.0f;
    float img_center_x = orig_img.cols / 2.0f;
    float img_center_y = orig_img.rows / 2.0f;
    float field_of_view_x = 90.0f;
    float field_of_view_y = 70.0f;
    float focal_length_x = img_center_x / std::tan((field_of_view_x / 2.0f) * M_PI / 180.0f);
    float focal_length_y = img_center_y / std::tan((field_of_view_y / 2.0f) * M_PI / 180.0f);
    lateral_angle = std::atan((center_x - img_center_x) / focal_length_x) * 180.0f / M_PI;
    vertical_angle = std::atan((center_y - img_center_y) / focal_length_y) * 180.0f / M_PI;
}

/**
 * @brief Converts a depth value to an estimated distance.
 */
float calculateDistance(float depth_value)
{
    float scaling_factor = 10.0f; // Example factor (needs calibration)
    return depth_value * scaling_factor;
}

/**
 * @brief Processes the frame through both models concurrently and combines the results.
 */
ModelResults processWithBothModels(int camera_id)
{
    ModelResults results;
    cv::Mat frame = takeSnapshotFromSelectedCamera(camera_id);
    if (frame.empty())
    {
        std::cerr << "Failed to capture image from camera " << camera_id << "\n";
        return results;
    }
    captured_image = frame.clone();

    std::vector<int> yolo_class_ids;
    std::vector<float> yolo_confidences;
    std::vector<cv::Rect> yolo_bboxes;
    std::vector<std::vector<float>> depth_map;

    // Run YOLO and depth estimation concurrently.
    std::thread yolo_thread([&]()
                            { yolo_bboxes = runYolo(yolo_class_ids, yolo_confidences, frame); });
    std::thread depth_thread([&]()
                             { depth_map = runDepthAnything(frame); });
    yolo_thread.join();
    depth_thread.join();

    if (yolo_bboxes.empty() || depth_map.empty())
    {
        return results;
    }

    float scale_x = float(depth_map[0].size()) / frame.cols;
    float scale_y = float(depth_map.size()) / frame.rows;

    for (size_t i = 0; i < yolo_bboxes.size(); i++)
    {
        DetectedObject obj;
        obj.objectClass = (yolo_class_ids[i] < YOLO_CLASSES.size()) ? YOLO_CLASSES[yolo_class_ids[i]] : "unknown";
        obj.bbox = yolo_bboxes[i];
        obj.confidence = yolo_confidences[i];

        int center_x = yolo_bboxes[i].x + yolo_bboxes[i].width / 2;
        int center_y = yolo_bboxes[i].y + yolo_bboxes[i].height / 2;
        int depth_x = std::max(0, std::min(int(center_x * scale_x), int(depth_map[0].size()) - 1));
        int depth_y = std::max(0, std::min(int(center_y * scale_y), int(depth_map.size()) - 1));
        float depth_value = depth_map[depth_y][depth_x];
        obj.objectDistance = calculateDistance(depth_value);
        calculateAngles(yolo_bboxes[i], frame, obj.objectLateralAngle, obj.objectVerticalAngle);
        results.objects.push_back(obj);
    }
    std::sort(results.objects.begin(), results.objects.end(), [](const DetectedObject &a, const DetectedObject &b)
              { return a.objectDistance < b.objectDistance; });
    return results;
}

/**
 * @brief Formats the distance value as a string.
 */
std::string formatDistance(float meters)
{
    char buffer[32];
    sprintf(buffer, "%.1f meters", meters);
    return std::string(buffer);
}

/**
 * @brief Main function setting up the HTTP server and endpoints.
 */
int main()
{
    std::cout << "Serving on Port 8008\n";
    std::cout << "Serving MJPEG Stream on Port 1189\n";

    cv::VideoCapture climberArmCap;
    cv::VideoCapture cap;
    httplib::Server svr;

    svr.Get("/", [](const httplib::Request &, httplib::Response &res)
            {
         std::string response = "Object Detection and Depth Estimation API using PyTorch C++ API";
         res.set_content(response, "text/plain"); });

    svr.Get("/get_closest_object", [](const httplib::Request &req, httplib::Response &res)
            {
         int cam_id = DEFAULT_CAMERA_DEVICE;
         if (req.has_param("camera"))
             cam_id = std::stoi(req.get_param_value("camera"));
         ModelResults results = processWithBothModels(cam_id);
         json response;
         response["Objects"] = json::array();
         if (!results.objects.empty()) {
             json object_data;
             const auto &obj = results.objects[0]; // Closest object
             object_data["objectClass"] = obj.objectClass;
             object_data["objectDistance"] = formatDistance(obj.objectDistance);
             object_data["objectLateralAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectLateralAngle));
             object_data["objectVerticalAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectVerticalAngle));
             response["Objects"].push_back(object_data);
         }
         res.set_content(response.dump(4), "application/json"); });

    svr.Get("/get_all_objects", [](const httplib::Request &req, httplib::Response &res)
            {
         
         int cam_id = DEFAULT_CAMERA_DEVICE;
         if (req.has_param("camera"))
             cam_id = std::stoi(req.get_param_value("camera"));
         ModelResults results = processWithBothModels(cam_id);
         json response;
         response["Objects"] = json::array();
         for (const auto &obj : results.objects) {
             json object_data;
             object_data["objectClass"] = obj.objectClass;
             object_data["objectDistance"] = formatDistance(obj.objectDistance);
             object_data["objectLateralAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectLateralAngle));
             object_data["objectVerticalAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectVerticalAngle));
             response["Objects"].push_back(object_data);
         }
         res.set_content(response.dump(4), "application/json"); });

    // New endpoint to get all YOLO detections (class and confidence only)
    svr.Get("/get_yolo_all_objects", [](const httplib::Request &req, httplib::Response &res)
            {
    int cam_id = DEFAULT_CAMERA_DEVICE;
    if (req.has_param("camera"))
        cam_id = std::stoi(req.get_param_value("camera"));

    cv::Mat frame = takeSnapshotFromSelectedCamera(cam_id);
    if (frame.empty()) {
        res.status = 500;
        res.set_content("{\"error\": \"Failed to capture image from camera\"}", "application/json");
        return;
    }
    
    std::vector<int> yolo_class_ids;
    std::vector<float> yolo_confidences;
    // Run YOLOv5 model only (depth estimation is not used here)
    std::vector<cv::Rect> yolo_bboxes = runYolo(yolo_class_ids, yolo_confidences, frame);
    
    json response;
    response["Objects"] = json::array();
    for (size_t i = 0; i < yolo_bboxes.size(); i++) {
        json object_data;
        object_data["objectClass"] = (yolo_class_ids[i] < YOLO_CLASSES.size()) ? YOLO_CLASSES[yolo_class_ids[i]] : "unknown";
        object_data["confidence"] = yolo_confidences[i];
        response["Objects"].push_back(object_data);
    }
    res.set_content(response.dump(4), "application/json"); });

    // New endpoint to get the highest-confidence YOLO detection only (class and confidence)
    svr.Get("/get_yolo_closest_object", [](const httplib::Request &req, httplib::Response &res)
            {
    int cam_id = DEFAULT_CAMERA_DEVICE;
    if (req.has_param("camera"))
        cam_id = std::stoi(req.get_param_value("camera"));

    cv::Mat frame = takeSnapshotFromSelectedCamera(cam_id);
    if (frame.empty()) {
        res.status = 500;
        res.set_content("{\"error\": \"Failed to capture image from camera\"}", "application/json");
        return;
    }
    
    std::vector<int> yolo_class_ids;
    std::vector<float> yolo_confidences;
    std::vector<cv::Rect> yolo_bboxes = runYolo(yolo_class_ids, yolo_confidences, frame);
    
    json response;
    response["Objects"] = json::array();
    if (!yolo_bboxes.empty()) {
        // Choose the detection with the highest confidence.
        size_t best_index = 0;
        float best_conf = yolo_confidences[0];
        for (size_t i = 1; i < yolo_confidences.size(); i++) {
            if (yolo_confidences[i] > best_conf) {
                best_conf = yolo_confidences[i];
                best_index = i;
            }
        }
        json object_data;
        object_data["objectClass"] = (yolo_class_ids[best_index] < YOLO_CLASSES.size()) ? YOLO_CLASSES[yolo_class_ids[best_index]] : "unknown";
        object_data["confidence"] = yolo_confidences[best_index];
        response["Objects"].push_back(object_data);
    }
    res.set_content(response.dump(4), "application/json"); });

    std::thread http_server_thread([&svr]()
                                   { svr.listen("0.0.0.0", 8008); });

    // ----- Open the video capture devices with error handling -----
    try {
        cap.open(DEFAULT_CAMERA_DEVICE);
    } catch(const cv::Exception &e) {
        std::cerr << "Exception while opening elevator camera: " << e.what() << std::endl;
    }
    if (!cap.isOpened()) {
        std::cerr << "Failed to open elevator camera. Using error image." << std::endl;
        frame = cv::imread("/home/ubuntu/deploy/bin/camera_error.png");
    }

    try {
        climberArmCap.open(1);
    } catch(const cv::Exception &e) {
        std::cerr << "Exception while opening climber arm camera: " << e.what() << std::endl;
    }
    if (!climberArmCap.isOpened()) {
        std::cerr << "Failed to open climber arm camera. Using error image." << std::endl;
        frame_climber_arm = cv::imread("/home/ubuntu/deploy/bin/camera_error.png");
    }
    
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 20};

    // ----- Start MJPEG streamers -----
    MJPEGStreamer streamer;
    streamer.start(1189);

    //MJPEGStreamer climberArmStreamer;
    //climberArmStreamer.start(1188);

    // ----- Streaming loop for climber arm camera -----
    // while(climberArmStreamer.isRunning()){
    //     try {
    //         climberArmCap >> frame_climber_arm;
    //     } catch(const cv::Exception &e) {
    //         std::cerr << "Exception during climber arm frame capture: " << e.what() << std::endl;
    //         frame_climber_arm = cv::imread("camera_error.png");
    //     }
    //     if (frame_climber_arm.empty()) {
    //         frame_climber_arm = cv::imread("camera_error.png");
    //     }
    //     std::vector<uchar> buff_bgr;
    //     cv::imencode(".jpg", frame_climber_arm, buff_bgr, params);
    //     climberArmStreamer.publish("/climber_camera", std::string(buff_bgr.begin(), buff_bgr.end()));
 
    //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // }

    // ----- Main streaming loop for elevator camera -----
    while (streamer.isRunning()) {
        try {
            cap >> frame;
        } catch(const cv::Exception &e) {
            std::cerr << "Exception during elevator camera frame capture: " << e.what() << std::endl;
            frame = cv::imread("/home/ubuntu/deploy/bin/camera_error.png");
        }
        if (frame.empty()) {
            frame = cv::imread("/home/ubuntu/deploy/bin/camera_error.png");
        }
        try {
            climberArmCap >> frame_climber_arm;
        } catch(const cv::Exception &e) {
            std::cerr << "Exception during climber arm frame capture: " << e.what() << std::endl;
            frame_climber_arm = cv::imread("/home/ubuntu/deploy/bin/camera_error.png");
        }
        if (frame_climber_arm.empty()) {
            frame_climber_arm = cv::imread("/home/ubuntu/deploy/bin/camera_error.png");
        }
        std::vector<uchar> buff_bgr;
        std::vector<uchar> buff_bgr_climber;
        cv::imencode(".jpg", frame_climber_arm, buff_bgr_climber, params);
        cv::imencode(".jpg", frame, buff_bgr, params);
        streamer.publish("/elevator_camera", std::string(buff_bgr.begin(), buff_bgr.end()));
        streamer.publish("/climber_camera", std::string(buff_bgr_climber.begin(), buff_bgr_climber.end()));

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    streamer.stop();
    svr.stop();
    http_server_thread.join();
    return 0;
}

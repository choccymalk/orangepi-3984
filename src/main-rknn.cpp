/**
 * @file main.cpp
 * @brief Object Detection and Depth Estimation Server using RKNN models (Multithreaded Version)
 *
 * This file implements an HTTP server that provides object detection (using YOLOv5)
 * and depth estimation (using Depth Anything) capabilities. The two inference tasks
 * run concurrently on the same captured frame using multithreading.
 *
 * Key components:
 * - YOLOv5 for object detection
 * - Depth Anything model for depth estimation
 * - HTTP endpoints for querying detection results
 * - Multithreaded model inference for improved throughput
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread> // For multithreading
#include "opencv2/opencv.hpp"
#include "httplib.h"
#include "rknn_api.h"
#include "json.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include "mjpeg_streamer.hpp"

#define HAVE_OPENCV_DNN
#define HAVE_OPENCV_FLANN
#define HAVE_OPENCV_HIGHGUI
#define HAVE_OPENCV_VIDEOIO
#define HAVE_OPENCV_VIDEO

using json = nlohmann::json;
using MJPEGStreamer = nadjieb::MJPEGStreamer;

// Depth Anything v2 model details:
// Input:
// Name: norm_tensor:0, tensor: NCHW, float16[1,3,518,518]
// Output:
// Name: norm_tensor:1, tensor: float16[1,518,518]
// yolov5 model details:
// Input: 
// Name: norm_tensor:0, tensor: NCHW, int8[1,3,640,640]
// Output:
// Name: norm_tensor:1, tensor: int8[1,255,80,80]
// Name: norm_tensor:2, tensor: int8[1,255,40,40]
// Name: norm_tensor:3, tensor: int8[1,255,20,20]

#define DEPTH_ANYTHING_MODEL_PATH "/home/ubuntu/deploy/models/depth-anything.rknn"
#define YOLO_MODEL_PATH "/home/ubuntu/deploy/models/yolov5.rknn"
#define DEFAULT_IMG_PATH "../test.jpg"
#define DEFAULT_CAMERA_DEVICE 0

cv::Mat captured_image;
std::mutex mtx;
cv::Mat frame;
cv::Mat frame_climber_arm;

/**
 * @struct DetectedObject
 * @brief Represents a detected object with spatial information
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
 * @brief Container for combined detection and depth estimation results
 */
struct ModelResults
{
    std::vector<DetectedObject> objects;
};

// YOLO classes (modify according to your model)
const std::vector<std::string> YOLO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

/**
 * @brief Loads model data from file
 * @param filename Path to RKNN model file
 * @param model_size Output parameter for model data size
 * @return Pointer to loaded model data, NULL on failure
 */
static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    unsigned char *data = (unsigned char *)malloc(size);
    if (data == NULL)
    {
        printf("Malloc buffer for model failed.\n");
        fclose(fp);
        return NULL;
    }

    fseek(fp, 0, SEEK_SET);
    int ret = fread(data, 1, size, fp);
    if (ret != size)
    {
        printf("Read model file failed.\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    *model_size = size;
    return data;
}

/**
 * @brief Captures a frame from specified camera device
 * @param camera_id Camera device ID (typically 0 for default camera)
 * @return Captured frame as cv::Mat, empty Mat on failure
 */
cv::Mat takeSnapshotFromSelectedCamera(const int camera_id)
{
    /*cv::VideoCapture camera(camera_id);
    if (!camera.isOpened())
    {
        std::cerr << "Could not open camera " << camera_id << std::endl;
        return cv::Mat();
    }
    cv::Mat frame;
    camera >> frame;
    if (frame.empty())
    {
        std::cerr << "Could not grab frame from camera " << camera_id << std::endl;
        return cv::Mat();
    }*/
    return frame;
}

/**
 * @brief Preprocesses image for model input
 * @param frame Original captured frame
 * @param target_width Target width for resizing
 * @param target_height Target height for resizing
 * @return Preprocessed image in RGB format
 */
cv::Mat loadAndPreprocessImage(const cv::Mat &frame, int target_width, int target_height)
{
    cv::Mat img;
    cv::resize(frame, img, cv::Size(target_width, target_height));

    // Convert to RGB (OpenCV loads as BGR)
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat img_uint8;
    img.convertTo(img_uint8, CV_8U);
    captured_image = frame.clone(); // Store original frame globally if needed
    return img_uint8;
}

/**
 * @brief Runs YOLOv5 object detection on the provided frame
 * @param[out] class_ids Vector to store detected class IDs
 * @param[out] confidences Vector to store detection confidence values
 * @param frame Input frame to process
 * @return Vector of detected bounding boxes
 */
std::vector<cv::Rect> runYolo(std::vector<int> &class_ids, std::vector<float> &confidences, const cv::Mat &frame)
{
    int ret;
    rknn_context ctx;
    std::vector<cv::Rect> bboxes;

    // 1. Load RKNN model using the method from the second file
    printf("Loading YOLO model...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(YOLO_MODEL_PATH, &model_data_size);
    if (!model_data)
    {
        printf("Load YOLO model failed.\n");
        return bboxes;
    }

    // 2. Initialize RKNN context
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init YOLO error ret=%d\n", ret);
        free(model_data);
        return bboxes;
    }

    // Set core mask if needed
    rknn_core_mask core_mask = RKNN_NPU_CORE_2;
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_set_core_mask error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return bboxes;
    }

    // 3. Get model input/output info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query RKNN_QUERY_IN_OUT_NUM error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return bboxes;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    //printf(std::to_string(input_attrs[io_num.n_input]));
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query RKNN_QUERY_INPUT_ATTR error ret=%d\n", ret);
            rknn_destroy(ctx);
            free(model_data);
            return bboxes;
        }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query RKNN_QUERY_OUTPUT_ATTR error ret=%d\n", ret);
            rknn_destroy(ctx);
            free(model_data);
            return bboxes;
        }
    }

    // 4. Use the provided frame instead of capturing a new one
    cv::Mat orig_frame = frame;
    int input_width = input_attrs[0].dims[2];
    int input_height = input_attrs[0].dims[1];
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        input_height = input_attrs[0].dims[2];
        input_width = input_attrs[0].dims[3];
    }

    cv::Mat img = loadAndPreprocessImage(orig_frame, input_width, input_height);
    if (img.empty())
    {
        printf("Failed to preprocess image.\n");
        rknn_destroy(ctx);
        free(model_data);
        return bboxes;
    }
    
    // Inside runYolo function after preprocessing the image
    cv::Mat chw_img;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        // Convert HWC to CHW
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        cv::merge(channels, chw_img); // Reorders to CHW
        img = chw_img;
    }
    //cv::bitwise_not(img, img);
    cv::imwrite("image.jpg", img);
    // 5. Prepare input tensor
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = input_attrs[0].fmt;
    inputs[0].buf = img.data;
    inputs[0].size = img.total() * img.channels();

    // 6. Run inference
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return bboxes;
    }

    ret = rknn_run(ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return bboxes;
    }

    // 7. Get output data
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }

    // write raw model output to file for debugging
    

    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, nullptr);
    if (ret < 0)
    {
        printf("rknn_outputs_get error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return bboxes;
    }

    // 8. Process YOLO output
    float *output_data = (float *)outputs[0].buf;
    int num_boxes = output_attrs[0].dims[1];
    int box_info_size = output_attrs[0].dims[2];
    float conf_threshold = 0.7; // Adjust as needed

    // Original image dimensions (for scaling bounding boxes)
    cv::Mat orig_img = captured_image;
    float scale_x = float(orig_img.cols) / input_width;
    float scale_y = float(orig_img.rows) / input_height;

    for (int i = 0; i < num_boxes; i++)
    {
        float *box_data = output_data + i * box_info_size;
        float box_conf = box_data[4];

        if (box_conf > conf_threshold)
        {
            int max_class_id = 0;
            float max_class_prob = 0;
            for (int j = 5; j < box_info_size; j++)
            {
                if (box_data[j] > max_class_prob)
                {
                    max_class_prob = box_data[j];
                    max_class_id = j - 5;
                }
            }

            float x = box_data[0] * scale_x;
            float y = box_data[1] * scale_y;
            float w = box_data[2] * scale_x;
            float h = box_data[3] * scale_y;

            int x1 = std::max(0, int(x - w / 2));
            int y1 = std::max(0, int(y - h / 2));
            int width = int(w);
            int height = int(h);

            bboxes.push_back(cv::Rect(x1, y1, width, height));
            class_ids.push_back(max_class_id);
            confidences.push_back(box_conf);
        }
    }

    // 9. Release resources
    for (int i = 0; i < io_num.n_output; i++)
    {
        rknn_outputs_release(ctx, 1, &outputs[i]);
    }
    rknn_destroy(ctx);
    free(model_data);

    return bboxes;
}

/**
 * @brief Runs Depth Anything model for depth estimation on the provided frame
 * @param frame Input frame to process
 * @return 2D vector representing depth map (row-major order)
 * @note Uses mutex lock for thread-safe model execution
 */
std::vector<std::vector<float>> runDepthAnything(const cv::Mat &frame)
{
    std::lock_guard<std::mutex> lock(mtx);
    int ret;
    rknn_context ctx;
    std::vector<std::vector<float>> depth_map;

    printf("Loading Depth Anything model...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(DEPTH_ANYTHING_MODEL_PATH, &model_data_size);
    if (!model_data)
    {
        printf("Load Depth Anything model failed.\n");
        return depth_map;
    }

    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init Depth Anything error ret=%d\n", ret);
        free(model_data);
        return depth_map;
    }

    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1; // Use a different core for the depth model
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_set_core_mask error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return depth_map;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query RKNN_QUERY_IN_OUT_NUM error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return depth_map;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query RKNN_QUERY_INPUT_ATTR error ret=%d\n", ret);
            rknn_destroy(ctx);
            free(model_data);
            return depth_map;
        }
    }
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query RKNN_QUERY_OUTPUT_ATTR error ret=%d\n", ret);
            rknn_destroy(ctx);
            free(model_data);
            return depth_map;
        }
    }

    // Use the provided frame for inference
    cv::Mat orig_frame = frame;
    int input_width = input_attrs[0].dims[2];
    int input_height = input_attrs[0].dims[1];
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        input_height = input_attrs[0].dims[2];
        input_width = input_attrs[0].dims[3];
    }

    cv::Mat img = loadAndPreprocessImage(orig_frame, input_width, input_height);
    if (img.empty())
    {
        printf("Failed to preprocess image.\n");
        rknn_destroy(ctx);
        free(model_data);
        return depth_map;
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;
    inputs[0].size = img.total() * img.channels();

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return depth_map;
    }

    ret = rknn_run(ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return depth_map;
    }

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, nullptr);
    if (ret < 0)
    {
        printf("rknn_outputs_get error ret=%d\n", ret);
        rknn_destroy(ctx);
        free(model_data);
        return depth_map;
    }

    // 8. Process depth map output
    int output_index = 0; // Adjust if needed
    int height = output_attrs[output_index].dims[1];
    int width = output_attrs[output_index].dims[2];
    if (output_attrs[output_index].fmt == RKNN_TENSOR_NCHW)
    {
        height = output_attrs[output_index].dims[2];
        width = output_attrs[output_index].dims[3];
    }

    float *output_data = (float *)outputs[output_index].buf;

    depth_map.resize(height);
    for (int h = 0; h < height; h++)
    {
        depth_map[h].resize(width);
        for (int w = 0; w < width; w++)
        {
            int index = h * width + w;
            depth_map[h][w] = output_data[index];
        }
    }

    for (int i = 0; i < io_num.n_output; i++)
    {
        rknn_outputs_release(ctx, 1, &outputs[i]);
    }
    rknn_destroy(ctx);
    free(model_data);

    return depth_map;
}

/**
 * @brief Calculates object angles relative to frame center
 * @param bbox Object bounding box
 * @param orig_img Original captured image
 * @param[out] lateral_angle Calculated lateral angle (degrees)
 * @param[out] vertical_angle Calculated vertical angle (degrees)
 * @note Assumes 90° horizontal and 70° vertical FOV (adjust if needed)
 */
void calculateAngles(const cv::Rect &bbox, const cv::Mat &orig_img, float &lateral_angle, float &vertical_angle)
{
    float center_x = bbox.x + bbox.width / 2.0f;
    float center_y = bbox.y + bbox.height / 2.0f;

    float img_center_x = orig_img.cols / 2.0f;
    float img_center_y = orig_img.rows / 2.0f;

    float offset_x = (center_x - img_center_x) / img_center_x;
    float offset_y = (center_y - img_center_y) / img_center_y;

    float field_of_view_x = 90.0f;
    float field_of_view_y = 70.0f;

    float focal_length_x = img_center_x / std::tan((field_of_view_x / 2.0f) * M_PI / 180.0f);
    float focal_length_y = img_center_y / std::tan((field_of_view_y / 2.0f) * M_PI / 180.0f);

    lateral_angle = std::atan((center_x - img_center_x) / focal_length_x) * 180.0f / M_PI;
    vertical_angle = std::atan((center_y - img_center_y) / focal_length_y) * 180.0f / M_PI;
}

/**
 * @brief Converts depth value to estimated distance
 * @param depth_value Raw depth value from model
 * @return Estimated distance in meters
 * @note Requires calibration with real-world measurements
 */
float calculateDistance(float depth_value)
{
    float scaling_factor = 10.0f; // Example value, needs calibration
    return depth_value * scaling_factor;
}

/**
 * @brief Processes image through both models concurrently and combines results
 * @param camera_id Camera device ID for image capture
 * @return ModelResults containing all detected objects with spatial info
 * @details Captures a single frame and then runs YOLO and Depth Anything concurrently.
 */
ModelResults processWithBothModels(int camera_id)
{
    ModelResults results;

    // Capture a single frame
    cv::Mat frame = takeSnapshotFromSelectedCamera(camera_id);
    if (frame.empty())
    {
        std::cerr << "Failed to capture image from camera " << camera_id << std::endl;
        return results;
    }

    captured_image = frame.clone(); // Update global frame if needed

    // Variables to hold outputs from each model
    std::vector<int> yolo_class_ids;
    std::vector<float> yolo_confidences;
    std::vector<cv::Rect> yolo_bboxes;
    std::vector<std::vector<float>> depth_map;

    // Launch threads for concurrent inference
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

    // Map depth map dimensions to the original image
    float scale_x = float(depth_map[0].size()) / frame.cols;
    float scale_y = float(depth_map.size()) / frame.rows;

    // Combine detections with corresponding depth estimates
    for (size_t i = 0; i < yolo_bboxes.size(); i++)
    {
        DetectedObject obj;
        obj.objectClass = (yolo_class_ids[i] < YOLO_CLASSES.size()) ? YOLO_CLASSES[yolo_class_ids[i]] : "unknown";
        obj.bbox = yolo_bboxes[i];
        obj.confidence = yolo_confidences[i];

        int center_x = yolo_bboxes[i].x + yolo_bboxes[i].width / 2;
        int center_y = yolo_bboxes[i].y + yolo_bboxes[i].height / 2;

        int depth_x = int(center_x * scale_x);
        int depth_y = int(center_y * scale_y);
        depth_x = std::max(0, std::min(depth_x, int(depth_map[0].size()) - 1));
        depth_y = std::max(0, std::min(depth_y, int(depth_map.size()) - 1));

        float depth_value = depth_map[depth_y][depth_x];
        obj.objectDistance = calculateDistance(depth_value);
        calculateAngles(yolo_bboxes[i], frame, obj.objectLateralAngle, obj.objectVerticalAngle);

        results.objects.push_back(obj);
    }

    std::sort(results.objects.begin(), results.objects.end(),
              [](const DetectedObject &a, const DetectedObject &b)
              {
                  return a.objectDistance < b.objectDistance;
              });

    return results;
}

/**
 * @brief Formats distance value for human-readable output
 * @param meters Distance value in meters
 * @return Formatted string with one decimal place
 */
std::string formatDistance(float meters)
{
    char buffer[32];
    sprintf(buffer, "%.1f meters", meters);
    return std::string(buffer);
}

/**
 * @brief Main function setting up HTTP server and endpoints
 * @details Exposes endpoints:
 * - /get_closest_object: Returns closest detected object
 * - /get_all_objects: Returns all detected objects sorted by distance
 */
int main()
{
    printf("Serving On Port 8008\n");
    printf("Serving MJPEG Stream On Port 1189 and 1188\n");

    cv::VideoCapture climberArmCap;
    cv::VideoCapture cap;
    httplib::Server svr;

    // Setup HTTP server routes
    svr.Get("/", [](const httplib::Request &, httplib::Response &res)
    {
        std::string responseToRequest = "Object Detection and Depth Estimation API";
        printf("GET /\n");
        res.set_content(responseToRequest, "text/plain"); 
    });

    svr.Get("/get_closest_object", [](const httplib::Request &req, httplib::Response &res)
    {
        printf("GET /get_closest_object\n");
        std::string cameraId_as_string = std::to_string(DEFAULT_CAMERA_DEVICE);
        if (req.has_param("camera")) {
            cameraId_as_string = req.get_param_value("camera");
        }
        int cameraId_as_int = std::stoi(cameraId_as_string);
        
        ModelResults results = processWithBothModels(cameraId_as_int);
        
        json response;
        response["Objects"] = json::array();
        
        if (!results.objects.empty()) {
            json object_data;
            const auto& obj = results.objects[0]; // Closest object
            object_data["objectClass"] = obj.objectClass;
            object_data["objectDistance"] = formatDistance(obj.objectDistance);
            object_data["objectLateralAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectLateralAngle));
            object_data["objectVerticalAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectVerticalAngle));
            response["Objects"].push_back(object_data);
        }
        res.set_content(response.dump(4), "application/json"); 
    });

    svr.Get("/get_all_objects", [](const httplib::Request &req, httplib::Response &res)
    {
        printf("GET /get_all_objects\n");
        std::string cameraId_as_string = std::to_string(DEFAULT_CAMERA_DEVICE);
        if (req.has_param("camera")) {
            cameraId_as_string = req.get_param_value("camera");
        }
        int cameraId_as_int = std::stoi(cameraId_as_string);
        
        ModelResults results = processWithBothModels(cameraId_as_int);
        
        json response;
        response["Objects"] = json::array();
        
        for (const auto& obj : results.objects) {
            json object_data;
            object_data["objectClass"] = obj.objectClass;
            object_data["objectDistance"] = formatDistance(obj.objectDistance);
            object_data["objectLateralAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectLateralAngle));
            object_data["objectVerticalAngleRelativeToCenterOfFrame"] = std::to_string(int(obj.objectVerticalAngle));
            response["Objects"].push_back(object_data);
        }
        res.set_content(response.dump(4), "application/json"); 
    });

    // Start HTTP server in a separate thread
    std::thread http_server_thread([&svr]() {
        svr.listen("0.0.0.0", 8008);
    });

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

    // Cleanup
    //climberArmStreamer.stop();
    streamer.stop();
    svr.stop(); // Stop the HTTP server
    http_server_thread.join(); // Wait for the HTTP server thread to finish

    return 0;
}

// #include <onnxruntime_cxx_api.h>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "exchange_box.h"

static constexpr int INPUT_W = 640;      // Width of input
static constexpr int INPUT_H = 640;      // Height of input
static constexpr int NUM_CLASSES = 1;    // Number of classes
static constexpr int NUM_ANCHORS = 8400; // Number of infer results
static constexpr float MERGE_CONF_ERROR = 0.6;
static constexpr float MERGE_MIN_IOU = 0.9;

/**
 * @brief YOLO图像预处理函数
 * @param src 输入图像(BGR格式)
 * @param dst 输出blob数据(可直接输入网络)
 * @param img_size 目标尺寸(默认640x640)
 * @param scale 返回的缩放比例
 * @param pad 返回的填充像素数(x,y方向)
 */
static void yolo_preprocess(cv::Mat& src, cv::Mat& dst, cv::Size img_size = cv::Size(INPUT_W, INPUT_H)) 
{
    // 1. 颜色空间转换 BGR -> RGB
    cv::Mat rgb;
    float scale;
    cv::Point pad;
    cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);

    // 2. Letterbox处理（保持宽高比缩放）
    int h = rgb.rows;
    int w = rgb.cols;
    float r = std::min((float)img_size.width / w, (float)img_size.height / h);
    int new_w = (int)(w * r);
    int new_h = (int)(h * r);
    
    // 计算填充量
    pad.x = (img_size.width - new_w) / 2;
    pad.y = (img_size.height - new_h) / 2;
    scale = 1.0f / r;

    // 3. 缩放并填充
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat padded(img_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad.x, pad.y, new_w, new_h)));

    // 4. 归一化并转换为浮点型
    cv::Mat float_img;
    padded.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 5. 创建blob (NCHW格式)
    cv::dnn::blobFromImage(float_img, dst, 1.0, cv::Size(), 
                          cv::Scalar(), true, false);
}
// static cv::Mat letterbox(const cv::Mat &img, Eigen::Matrix3f &transform_matrix,
//                          std::vector<int> new_shape = {INPUT_W, INPUT_H})
// {
//     // Get current image shape [height, width]
//     int img_h = img.rows;
//     int img_w = img.cols;

//     // Compute scale ratio(new / old) and target resized shape
//     float scale =
//         std::min(new_shape[1] * 1.0 / img_h, new_shape[0] * 1.0 / img_w);
//     int resize_h = static_cast<int>(round(img_h * scale));
//     int resize_w = static_cast<int>(round(img_w * scale));

//     // Compute padding
//     int pad_h = new_shape[1] - resize_h;
//     int pad_w = new_shape[0] - resize_w;

//     // Resize and pad image while meeting stride-multiple constraints
//     cv::Mat resized_img;
//     cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

//     // Divide padding into 2 sides
//     float half_h = pad_h * 1.0 / 2;
//     float half_w = pad_w * 1.0 / 2;

//     // Compute padding boarder
//     int top = static_cast<int>(round(half_h - 0.1));
//     int bottom = static_cast<int>(round(half_h + 0.1));
//     int left = static_cast<int>(round(half_w - 0.1));
//     int right = static_cast<int>(round(half_w + 0.1));

//     /* clang-format off */
//     /* *INDENT-OFF* */

//     // Compute point transform_matrix
//     transform_matrix << 1.0 / scale, 0, -half_w / scale,
//     0, 1.0 / scale, -half_h / scale,
//     0, 0, 1;

//     /* *INDENT-ON* */
//     /* clang-format on */

//     // Add border
//     cv::copyMakeBorder(resized_img, resized_img, top, bottom, left, right,
//                        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

//     return resized_img;
// }

// Decode output tensor
static void generateProposals(
    std::vector<ExchangeBox> &output_objs, 
    const cv::Mat &output_buffer,
    const Eigen::Matrix<float, 3, 3> &transform_matrix, float conf_threshold)
{
    for (int anchor_idx = 0; anchor_idx < NUM_ANCHORS; anchor_idx++)
    {
        double class_score;
        cv::Point class_id;
        cv::Mat num_scores =
            output_buffer.col(anchor_idx);
        cv::waitKey(1);
    //             .rowRange(4,
    //                       4 + NUM_CLASSES);
    //     // Argmax
    //     cv::minMaxLoc(num_scores, NULL, &class_score, NULL, &class_id);
    //     cv::Point color_id = class_id;
    //     if (class_score < conf_threshold)
    //     {
    //         continue;
    //     }

    //     // Get keypoints
    //     float x_1 = output_buffer.at<float>(6, anchor_idx);
    //     float y_1 = output_buffer.at<float>(7, anchor_idx);
    //     float x_2 = output_buffer.at<float>(8, anchor_idx);
    //     float y_2 = output_buffer.at<float>(9, anchor_idx);
    //     float x_3 = output_buffer.at<float>(10, anchor_idx);
    //     float y_3 = output_buffer.at<float>(11, anchor_idx);
    //     float x_4 = output_buffer.at<float>(12, anchor_idx);
    //     float y_4 = output_buffer.at<float>(13, anchor_idx);
    //     float x_5 = output_buffer.at<float>(14, anchor_idx);
    //     float y_5 = output_buffer.at<float>(15, anchor_idx);

    //     // Transform keypoints to raw image
    //     Eigen::Matrix<float, 3, 5> apex_norm;
    //     Eigen::Matrix<float, 3, 5> apex_dst;
    //     /* clang-format off */
    // /* *INDENT-OFF* */
    // apex_norm << x_1, x_2, x_3, x_4, x_5,
    //             y_1, y_2, y_3, y_4, y_5,
    //             1,   1,   1,   1,   1;
    // /* *INDENT-ON* */
    //     /* clang-format on */
    //     apex_dst = transform_matrix * apex_norm;

    //     // Packed as RuneObject
    //     RuneObject obj;
    //     obj.pts.r_center = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
    //     obj.pts.bottom_left = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
    //     obj.pts.top_left = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
    //     obj.pts.top_right = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));
    //     obj.pts.bottom_right = cv::Point2f(apex_dst(0, 4), apex_dst(1, 4));

    //     auto rect = cv::boundingRect(obj.pts.toVector2f());

    //     obj.box = rect;
    //     obj.color = DNN_COLOR_TO_ENEMY_COLOR[color_id.x];
    //     obj.prob = class_score;

    //     rects.push_back(rect);
    //     scores.push_back(class_score);
    //     output_objs.push_back(std::move(obj));
    }
}

int main()
{
    //     auto env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "YOLOV11");
    //     std::string weightFile = "/home/lixinhao/Project/yolo_pose_detect/data/model/best.onnx";

    //     // 将 std::string 转换为 ORTCHAR_T
    //     ORTCHAR_T* ort_weightFile;
    // #ifdef _WIN32
    //     {
    //         std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    //         std::wstring wide = converter.from_bytes(weightFile);
    //         ort_weightFile = const_cast<ORTCHAR_T*>(wide.c_str());
    //     }
    // #else
    //     ort_weightFile = const_cast<ORTCHAR_T*>(weightFile.c_str());
    // #endif

    //     Ort::SessionOptions session_options;
    //     session_options.SetIntraOpNumThreads(1);
    //     // 设置图优化级别为全部优化（最大优化）
    //     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    //     // 使用CPU版本的Session构造函数
    //     try
    //     {
    //         Ort::Session session(env, ort_weightFile, session_options);
    //         static constexpr const int width_ = 640; // 模型input width
    //         static constexpr const int height_ = 640; // 模型input height
    //         Ort::Value input_tensor_{nullptr};

    //         std::array<int64_t, 4> input_shape_{1, 3, height_, width_}; // NCHW, 1x3xHxW
    //         Ort::Value output_tensor_{nullptr};
    //         std::array<int64_t, 3> output_shape_{1, 13, 8400}; // 模型output shape，此处假设是二维的(1,3)

    //         std::array<float, width_ * height_ * 3> input_image_{}; // 输入图片，HWC
    //         std::array<float, 109200> results_{}; // 模型输出，注意和output_shape_对应

    //         std::string imgPath = "data/pictures/20250331_202850_FNmE13Sw.jpg";
    //         cv::Mat img = cv::imread(imgPath);

    //         auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    //         input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    //         output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
    //         const char* input_names[] = {"images"}; // 输入节点名
    //         const char* output_names[] = {"output0"}; // 输出节点名

    //         // 预处理
    //         cv::Mat img_f32;
    //         img.convertTo(img_f32, CV_32FC3); // 转float

    //         // BGR2RGB,
    //         for (int i = 0; i < img.rows; i++) {
    //             for (int j = 0; j < img.cols; j++) {
    //                 input_image_[i * img.cols + j + 0] = img_f32.at<cv::Vec3f>(i, j)[2];
    //                 input_image_[i * img.cols + j + 1 * img.cols * img.rows] = img_f32.at<cv::Vec3f>(i, j)[1];
    //                 input_image_[i * img.cols + j + 2 * img.cols * img.rows] = img_f32.at<cv::Vec3f>(i, j)[0];
    //             }
    //         }

    //         session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

    //         // 获取output的shape
    //         Ort::TensorTypeAndShapeInfo shape_info = output_tensor_.GetTensorTypeAndShapeInfo();

    //         // 获取output的dim
    //         size_t dim_count = shape_info.GetDimensionsCount();
    //         std::cout<< dim_count << std::endl;

    //         // 获取output的shape
    //         int64_t dims[3];
    //         shape_info.GetDimensions(dims, sizeof(dims) / sizeof(dims[0]));
    //         std::cout<< "output shape: " <<dims[0] << "," << dims[1] << "," << dims[2] <<std::endl;

    //         // 取output数据
    //         float* f = output_tensor_.GetTensorMutableData<float>();
    //         for(int i = 0; i < dims[1]; i++)
    //         {
    //             std::cout<< f[i]<< std::endl;
    //         }
    //     }
    //     catch(const Ort::Exception& e)
    //     {
    //         std::cerr << e.what() << '\n';
    //     }
    //     catch(const std::exception& e)
    //     {
    //         std::cerr << e.what() << '\n';
    //     }
    //     catch(const cv::Exception& e)
    //     {
    //         std::cerr << e.what() << '\n';
    //     }

    std::unique_ptr<ov::Core> ov_core_;
    std::unique_ptr<ov::CompiledModel> compiled_model_;
    const std::filesystem::path model_path_ = "/home/lixinhao/Project/yolo_pose_detect/data/model/best_openvino_model/best.xml";
    const std::string device_name_ = "CPU";
    float conf_threshold_ = 0.5f;
    int top_k = 10;
    float nms_threshold = 0.45f;

    ov_core_ = std::make_unique<ov::Core>();
    auto model = ov_core_->read_model(model_path_);

    // Set infer type
    ov::preprocess::PrePostProcessor ppp(model);
    // Set input output precision
    auto elem_type = device_name_ == "GPU" ? ov::element::f16 : ov::element::f32;
    auto perf_mode =
        device_name_ == "GPU"
            ? ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)
            : ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY);
    ppp.input().tensor().set_element_type(elem_type);
    ppp.output().tensor().set_element_type(elem_type);

    // Compile model
    compiled_model_ = std::make_unique<ov::CompiledModel>(
        ov_core_->compile_model(model, device_name_, perf_mode));

    cv::Mat img = cv::imread("data/pictures/20250331_202850_FNmE13Sw.jpg");
    
    float scale_ = 1.0f;
    cv::Size resize = cv::Size(INPUT_W, INPUT_H);
    cv::Point2f pad_ = cv::Point2f(0, 0);
    cv::Mat blob = cv::Mat(INPUT_H, INPUT_W, CV_32FC3);
    yolo_preprocess(img, blob, resize);
    // Reprocess
    Eigen::Matrix3f
        transform_matrix; // transform matrix from resized image to source image.
    //cv::Mat resized_img = letterbox(rgb_img, transform_matrix);

    // BGR->RGB, u8(0-255)->f32(0.0-1.0), HWC->NCHW
    // note: TUP's model no need to normalize
    // cv::Mat blob = cv::dnn::blobFromImage(
    //     resized_img, 1. / 255., cv::Size(INPUT_W, INPUT_H), cv::Scalar(0, 0, 0), true);

    // Feed blob into input
    auto input_port = compiled_model_->input();
    ov::Tensor input_tensor(
        input_port.get_element_type(),
        ov::Shape(std::vector<size_t>{1, 3, INPUT_W, INPUT_H}), blob.ptr(0));

    // Start inference
    // Lock because of the thread race condition within the openvino library
    auto infer_request = compiled_model_->create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output = infer_request.get_output_tensor();

    // Process output data
    auto output_shape = output.get_shape();
    // 4725 x 15 Matrix
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F,
                          output.data());

    // Parsed variables
    // std::vector<RuneObject> objs_tmp, objs_result;
    std::vector<ExchangeBox> exchange_boxes;

    // Parse YOLO output
    generateProposals(exchange_boxes, output_buffer, transform_matrix,
                      conf_threshold_);

}

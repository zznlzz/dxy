#include <opencv2/opencv.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <math.h>
#include "yolo.h"
#include <iostream>
#include <cmath>
#include "global.h"
#include <string.h>

#define USE_CUDA true
#define ip_address  "192.168.159.182"

using namespace std;
using namespace cv;
using namespace dnn;

cv::Mat hough(cv::Mat src);

char coord[50];
char coord_last_frame[50];
int target_x = 320;
int target_y = 240;
int flag_servo = 0;
int count_servo = 0;
int count_nomove = 0;
Matrix3d K; // 内参矩阵
Vector2d D; // 畸变矩阵
double calculateDistance(double x1, double y1, double x2, double y2);
void drive_servo(char *str, int target_x, int target_y, int m);

int main()
{
    string model_path = "../models/bestv5.onnx";
    string model_path_circle = "../models/best_circle.onnx";

    cv::Mat K = (cv::Mat_<double>(3, 3) << 5.866604127618223e+02, 0, 3.091697495003905e+02,
                 0, 5.862334531989521e+02, 2.301569065668424e+02,
                 0, 0, 1);

    cv::Mat D = (cv::Mat_<double>(4, 1) << 0.108035628286270, -0.264954789302431, 0, 0);

    coord[0] = '0'; // 字符串赋初值，不然发送数据会阻塞

    Yolo test;
    Net net1, net2;
    if (test.readModel(net1, model_path, USE_CUDA))
    {
        cout << "read net1 ok!" << endl;
    }
    else
    {
        cout << "read onnx1 model failed!";
        return -1;
    }

    if (test.readModel(net2, model_path_circle, USE_CUDA))
    {
        cout << "read net2 ok!" << endl;
    }
    else
    {
        cout << "read onnx2 model failed!";
        return -1;
    }

    // 生成随机颜色
    vector<Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++)
    {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }

    vector<Output> result;

    // 创建套接字
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    // 设置服务器地址
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8000);
    server_addr.sin_addr.s_addr = inet_addr(ip_address);

    // 连接到服务器
    connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr));

    // 接收图像数据
    while (true)
    {
        // 接收图像大小
        int size;
        char size_buf[16] = {0};
        recv(sock, size_buf, 16, 0);
        size = atoi(size_buf);

        // 接收图像数据
        char data_buf[size];
        int data_received = 0;
        while (data_received < size)
        {
            int ret = recv(sock, data_buf + data_received, size - data_received, 0);
            if (ret == -1)
            {
                break;
            }
            int n;
            data_received += ret;
        }

        // 将图像数据转换成图像格式
        Mat src = imdecode(Mat(1, size, CV_8UC1, data_buf), IMREAD_COLOR);
        Mat img;

        // 图像去畸变
        undistort(src, img, K, D, K);

        if (test.Detect(img, net1, result))
        {
            img = test.drawPred(img, result, color);
        }
        if (test.Detect(img, net2, result))
        {
            img = test.drawPred(img, result, color);
        }
        imshow("frame", img);

        drive_servo(coord, target_x, target_y, 20);

        if(strcmp(coord, coord_last_frame) == 0){ // 如果连续多帧坐标相同，说明没有检测到目标，不再发送坐标数据
            count_nomove++;
            if (count_nomove >= 20){
                strcpy(coord, "0");
                count_nomove = 0;
            }
        }
        send(sock, coord, strlen(coord), 0);
        strcpy(coord_last_frame, coord);
        if (waitKey(1) == 'q')
        {
            break;
        }
    }

    // 关闭连接
    close(sock);

    return 0;
}

void drive_servo(char *str, int target_x, int target_y, int times)
{
    int x, y, r;
    int rmax = 50;
    if (sscanf(str, "%d,%d", &x, &y) == 2)
    {
        r = calculateDistance(x, y, target_x, target_y);
        
        if (r <= rmax)
            count_servo++;
        else
            count_servo = 0;
        cout << r << "|" << count_servo << endl;
        if (count_servo >= times)
            flag_servo = 1;
    }
}

// 计算两点间距离
double calculateDistance(double x1, double y1, double x2, double y2)
{
    double dx = x2 - x1;
    double dy = y2 - y1;
    double distance = std::sqrt(dx * dx + dy * dy);
    return distance;
}

// cv::Mat hough(cv::Mat src)
// {
//     extern int flag;
//     cv::Mat dst, out;
//     // cv::medianBlur(src, dst, 3);
//     cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY); // 改为灰度图
//     // cv::GaussianBlur(dst, dst, cv::Size(9, 9), 2, 2);
//     cv::bilateralFilter(dst, out, 3, 100, 100);
//     std::vector<cv::Vec3f> circles;
//     cv::HoughCircles(out, circles, cv::HOUGH_GRADIENT_ALT, 1.5, 500, 300, 0.8); // 霍夫圆检测
//     // dp-累加分辨率大小-默认为1   两圆心之间最小距离   Canny边缘检测高阈值-低阈值自动为高阈值一半
//     // 越大检测的圆越接近完美圆形   圆半径最小值   圆半径最大值
//     // dp值越大，累加器分辨率越低，运行速度越快
//     flag = 0;
//     for (int i = 0; i < circles.size(); i++)
//     {
//         flag = 1;
//         cv::Vec3f c = circles[i];
//         cv::circle(src, cv::Point(c[0], c[1]), c[2], cv::Scalar(0, 255, 255), 3, cv::LINE_AA); // 圆周
//         cv::circle(src, cv::Point(c[0], c[1]), 2, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);      // 圆心
//         // std::cout << "x = " << c[0] << "y = " << c[1] << std::endl;
//         strcpy(coord, getUndistortedPixelCoord(c[0], c[1]));
//     }
//     return src;
// }

// char *getUndistortedPixelCoord(double x, double y)
// {
//     Eigen::Vector3d p;
//     p << x, y, 1;

//     // Eigen::Vector3d p_norm = K.inverse() * Eigen::Vector3d(p(0), p(1), 1);
//     Eigen::Vector3d p_norm = K * p;
//     double r2 = p_norm(0) * p_norm(0) + p_norm(1) * p_norm(1);
//     Eigen::Vector2d p_undistorted = p_norm.head<2>() * (1 + r2 * D(0)) + Eigen::Vector2d(2 * D(1) * p_norm(0) * p_norm(1), D(0) * (r2 + 2 * p_norm(0) * p_norm(0)));

//     // Eigen::Vector3d p_pixel = K * Eigen::Vector3d(p_undistorted(0), p_undistorted(1), 1);
//     Eigen::Vector3d p_pixel = K.inverse() * Eigen::Vector3d(p_undistorted(0), p_undistorted(1), 1);
//     Eigen::Vector2d p_actual_pixel(p_pixel(0) / p_pixel(2), p_pixel(1) / p_pixel(2));

//     char *str = new char[50];
//     sprintf(str, "%.6f,%.6f", p_actual_pixel(0), p_actual_pixel(1));
//     return str;
// }
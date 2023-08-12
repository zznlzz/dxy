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
#include <ctime>

#define USE_CUDA true
#define ip_address "192.168.10.101"

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
int model_flag = 0;
Matrix3d K; // 内参矩阵
Vector2d D; // 畸变矩阵
double calculateDistance(double x1, double y1, double x2, double y2);
void drive_servo(char *str, int target_x, int target_y, int m);

int main()
{
    string model_path = "../models/bestv5.onnx";
    string model_path_circle = "../models/best_circle.onnx";

    // 旧摄像头参数
    // cv::Mat K = (cv::Mat_<double>(3, 3) << 5.866604127618223e+02, 0, 3.091697495003905e+02,
    //              0, 5.862334531989521e+02, 2.301569065668424e+02,
    //              0, 0, 1);

    // cv::Mat D = (cv::Mat_<double>(4, 1) << 0.108035628286270, -0.264954789302431, 0, 0);

    // camerav2参数（不加偏振片）
    // cv::Mat K = (cv::Mat_<double>(3, 3) << 532.3404, 0, 319.3003,
    //                                         0, 532.4891, 257.4185,
    //                                         0, 0, 1);

    // cv::Mat D = (cv::Mat_<double>(4, 1) << 0.2156, -0.3761, 0, 0);

    // camerav2参数（加偏振片）
    //【我也想不明白为什么加了偏振片畸变变小了，应该是标定板不够平】
    cv::Mat K = (cv::Mat_<double>(3, 3) << 514.0045, 0, 321.6074,
                                            0, 514.6655, 260.0872,
                                            0, 0, 1);

    cv::Mat D = (cv::Mat_<double>(4, 1) << 0.1631, -0.2023, 0, 0);

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
    bool isFirstFrame = true;
    VideoWriter video;

    int num_frames = 200;
    time_t start,end;
    Mat frame;
    int i = 0;

    while(true)
    {
        // 接收图像大小
        if (i == 10) time(&start);//计时开始
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

        if (test.Detect(img, net1, result, 1))
        {
            img = test.drawPred(img, result, color, 1);
        }
        if (test.Detect(img, net2, result, 0))
        {
            img = test.drawPred(img, result, color, 0);
        }
        namedWindow("frame", WINDOW_NORMAL);
        imshow("frame", img);
        resizeWindow("frame", 1280, 960);

        drive_servo(coord, target_x, target_y, 60);

        if (strcmp(coord, coord_last_frame) == 0)
        { // 如果连续多帧坐标相同，说明没有检测到目标，不再发送坐标数据
            count_nomove++;
            if (count_nomove >= 30)
            {
                strcpy(coord, "0");
                count_nomove = 0;
            }
        }
        send(sock, coord, strlen(coord), 0);
        strcpy(coord_last_frame, coord);

        // 保存帧为视频
        if (isFirstFrame)
        {
            time_t now = time(0);
            tm *ltm = localtime(&now);
            char timestamp[20];
            sprintf(timestamp, "%04d-%02d-%02d_%02d:%02d:%02d", 1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
                    ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
            string videoFilename = "../video/" + string(timestamp) + ".avi";
            video.open(videoFilename, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(img.cols, img.rows));
            if (!video.isOpened())
            {
                cerr << "Error: Failed to create video file: " << videoFilename << endl;
                break;
            }
            isFirstFrame = false;
        }
        video.write(img);

        if (waitKey(1) == 'q' || waitKey(1) == 'Q')
        {
            break;
        }
        if(i == 10 + num_frames)
        {
            time(&end);//计时结束
            double fps,seconds = difftime (end,start);//计时
            fps = (num_frames-10) / seconds;
            cout << "Camera FPS is " << fps << endl;
        }
        i++;
    }

    // 关闭连接
    close(sock);

    // 释放视频资源
    video.release();

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
            model_flag = 1;
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

#include <opencv2/opencv.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <math.h>
#include "yolo.h"
#include <iostream>
#include <Eigen/Core> // 稠密矩阵代数运算（逆、特征值等）
#include <Eigen/Dense>
using namespace Eigen;

//#include "HoughCircles.cpp"
//#include "global.h"

#define USE_CUDA true

using namespace std;
using namespace cv;
using namespace dnn;

cv::Mat hough(cv::Mat src);
char coord[50] = {0};
int flag = 0;

int main() {
    extern int flag;
    string model_path = "/home/zyt/1dxy/yolov7-opencv-dnn-cpp-main/models/best.onnx";
    coord[0] = '0';
    Yolo test;
	Net net;
	if (test.readModel(net, model_path, USE_CUDA)) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}

	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
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
    server_addr.sin_addr.s_addr = inet_addr("192.168.43.169");

    // 连接到服务器
    connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr));

    // 接收图像数据
    while (true) {
        // 接收图像大小
        int size;
        char size_buf[16] = {0};
        recv(sock, size_buf, 16, 0);
        size = atoi(size_buf);

        // 接收图像数据
        char data_buf[size];
        int data_received = 0;
        while (data_received < size) {
            int ret = recv(sock, data_buf + data_received, size - data_received, 0);
            if (ret == -1) {
                break;
            }
            data_received += ret;
        }

        // 将图像数据转换成图像格式
        Mat img = imdecode(Mat(1, size, CV_8UC1, data_buf), IMREAD_COLOR);

        // 显示图像
        img = hough(img);
        // if (test.Detect(img, net, result)) {
		// 		img = test.drawPred(img, result, color);
		// 	}else {
		// 		cout << "Error: detect failed!" << endl;
		// 	}
        imshow("frame", img);

        //if (flag == 1){
        //send(sock, "0", 1, 0);
        send(sock, coord, strlen(coord), 0);
        //std::cout << "坐标数据已发送" << std::endl;
        //}

        if (waitKey(1) == 'q') {
            break;
        }
    }

    // 关闭连接
    close(sock);

    return 0;
}

cv::Mat hough(cv::Mat src)
{
    extern int flag;
    cv::Mat dst, out;    
    // cv::medianBlur(src, dst, 3);
    cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);// 改为灰度图
    // cv::GaussianBlur(dst, dst, cv::Size(9, 9), 2, 2);
    cv::bilateralFilter(dst, out, 3, 100, 100);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(out, circles, cv::HOUGH_GRADIENT_ALT, 1.5, 500, 300, 0.8);// 霍夫圆检测
    // dp-累加分辨率大小-默认为1   两圆心之间最小距离   Canny边缘检测高阈值-低阈值自动为高阈值一半
    // 越大检测的圆越接近完美圆形   圆半径最小值   圆半径最大值
    // dp值越大，累加器分辨率越低，运行速度越快
    flag = 0;
    for(int i=0; i<circles.size(); i++){
        flag = 1;
        cv::Vec3f c = circles[i];
        cv::circle(src, cv::Point(c[0],c[1]), c[2], cv::Scalar(0,255,255), 3, cv::LINE_AA);// 圆周
        cv::circle(src, cv::Point(c[0],c[1]), 2, cv::Scalar(255,0,0), 3, cv::LINE_AA);// 圆心
        // std::cout << "x = " << c[0] << "y = " << c[1] << std::endl;

        Matrix3d K; // 内参矩阵
        K <<   5.866604127618223e+02,            0,                  0, 
                            0,               5.862334531989521e+02,       0, 
                    3.091697495003905e+02,   2.301569065668424e+02,       1;
        
        // 畸变矩阵
        Vector2d D;
        D << 0.108035628286270, -0.264954789302431;

        // 像素坐标
        Vector2d p;
        p << c[0], c[1];

        // 归一化坐标
        Vector3d p_norm = K.inverse() * Vector3d(p(0), p(1), 1);

        // 去除畸变
        double r2 = p_norm(0) * p_norm(0) + p_norm(1) * p_norm(1);
        Vector2d p_undistorted = p_norm.head<2>() * (1 + r2 * D(0)) + Vector2d(2 * D(1) * p_norm(0) * p_norm(1), D(0) * (r2 + 2 * p_norm(0) * p_norm(0)));

        // 像素坐标
        Vector3d p_pixel = K * Vector3d(p_undistorted(0), p_undistorted(1), 1);

        Vector2d p_actual_pixel(p_pixel(0) / p_pixel(2), p_pixel(1) / p_pixel(2));
        std::cout << "去除畸变后的像素坐标：(" << p_actual_pixel(0) << ", " << p_actual_pixel(1) << ")" << std::endl;    

        char xx[20]={0};
        char yy[20]={0};
        sprintf(xx, "%.6f", p_actual_pixel(0));
        sprintf(yy, "%.6f", p_actual_pixel(1));
        strcpy(coord, xx);
        strcat(coord, ",");
        strcat(coord, yy);
    }  
    return src;
}
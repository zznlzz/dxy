#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
// #include "HoughCircles.hpp"
#include <Eigen/Core> // 稠密矩阵代数运算（逆、特征值等）
#include <Eigen/Dense>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

cv::Mat hough(cv::Mat src);
cv::Mat addImages(const cv::Mat& src1, const cv::Mat& src2);

Matrix3d K; // 内参矩阵
Vector2d D; // 畸变矩阵

int main(int argc,char** argv)
{
    K <<   5.866604127618223e+02,            0,                   0, 
                    0,               5.862334531989521e+02,       0, 
            3.091697495003905e+02,   2.301569065668424e+02,       1;
        
    D << 0.108035628286270, -0.264954789302431;
    
    //打开摄像头
    VideoCapture cap;//声明相机捕获对象
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    cap.set(CAP_PROP_FRAME_WIDTH,640);//图像的宽
    cap.set(CAP_PROP_FRAME_HEIGHT,480);//图像的高

    int deviceID = 0;//相机设备号
    // http://admin:admin@192.168.1.105:8081
    cap.open("http://admin:admin@192.168.1.102:8081");
    if (!cap.isOpened())
    {
        cerr << "Error: unable to open camera." << endl;
        return -1;
    }else{
        cout <<"Successfully turned on the camera."<< endl;
    }

    //输出图片

    Mat img;
    while (true)
    {
        cap >> img;
        namedWindow("camera",1);
        if (img.empty() == false)
        {
            img = hough(img);
            cv::imshow("camera",img);
        }else{
            cerr<<"Error: unable to read imagine." << endl;
        }
        
       if (waitKey(10) >= 0) break;
    }
    cap.release();
    destroyAllWindows();

    return 0;
}

cv::Mat hough(cv::Mat src)
{
    extern int flag;
    cv::Mat dst, out;
    cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
    cv::bilateralFilter(dst, out, 3, 100, 100);
    cv::Mat src2(dst.size(), dst.type(), cv::Scalar(10, 10, 10));

    // 设置颜色阈值
    cv::Scalar lower_white = cv::Scalar(200, 200, 200); // 白色的下限阈值
    cv::Scalar upper_white = cv::Scalar(255, 255, 255); // 白色的上限阈值
    
    // 颜色阈值化
    cv::Mat mask;
    cv::inRange(src, lower_white, upper_white, mask);
    
    // 应用颜色掩码
    cv::Mat masked;
    cv::bitwise_and(out, mask, masked);
    //cv::cvtColor(masked, masked, cv::COLOR_GRAY2BGR);
    //test = detectShapes(masked);
    masked = addImages(masked, src2);
    cv::imshow("mask", masked);

    std::vector<cv::Vec3f> circles;
    // cv::Mat edges;

    cv::HoughCircles(masked, circles, cv::HOUGH_GRADIENT_ALT, 1.5, 500, 300, 0.8);// 霍夫圆检测
    // cv::HoughCircles(masked, circles, cv::HOUGH_GRADIENT, 1, 500, 100, 0.8);// 霍夫圆检测
    // dp-累加分辨率大小-默认为1   两圆心之间最小距离   Canny边缘检测高阈值-低阈值自动为高阈值一半
    // 越大检测的圆越接近完美圆形   圆半径最小值   圆半径最大值
    // dp值越大，累加器分辨率越低，运行速度越快
    for(int i=0; i<circles.size(); i++){
        cv::Vec3f c = circles[i];
        cv::circle(src, cv::Point(c[0],c[1]), c[2], cv::Scalar(0,255,255), 3, cv::LINE_AA);// 圆周
        cv::circle(src, cv::Point(c[0],c[1]), 2, cv::Scalar(255,0,0), 3, cv::LINE_AA);// 圆心
        // std::cout << "x = " << c[0] << "y = " << c[1] << std::endl;

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
        std::cout << "(" << p_actual_pixel(0) << ", " << p_actual_pixel(1) << ")" << std::endl;   
        }  
    return src;
}

cv::Mat addImages(const cv::Mat& src1, const cv::Mat& src2) {
    // 确保两个图像具有相同的尺寸和类型
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());

    cv::Mat result;
    cv::Mat mask = (src1 > 0);  // 创建一个掩码，检查图像1中像素是否有内容

    // 使用条件操作，将图像1的像素中有内容的部分复制到结果中，没有内容的部分使用图像2的像素值
    cv::Mat maskedSrc1, maskedSrc2;
    src1.copyTo(maskedSrc1, mask);
    src2.copyTo(maskedSrc2, ~mask);  // ~ 表示按位取反

    // 将两个部分相加得到最终结果
    cv::add(maskedSrc1, maskedSrc2, result);

    return result;
}

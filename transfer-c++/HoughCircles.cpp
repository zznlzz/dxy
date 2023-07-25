#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include "global.h"
// using namespace cv;
// using namespace std;
cv::Mat hough(cv::Mat src)
{
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
    for(int i=0; i<circles.size(); i++){
        cv::Vec3f c = circles[i];
        cv::circle(src, cv::Point(c[0],c[1]), c[2], cv::Scalar(0,255,255), 3, cv::LINE_AA);// 圆周
        cv::circle(src, cv::Point(c[0],c[1]), 2, cv::Scalar(255,0,0), 3, cv::LINE_AA);// 圆心
        std::cout << "x = " << c[0] << " y = " << c[1] << std::endl;
        
        char * xx;
        char * yy;
        std::cout << 1 << std::endl ;
        sprintf(xx, "%.3f", c[0]);
        sprintf(yy, "%.3f", c[1]);
        std::cout << 2 << std::endl ;
        strcpy(coord, xx);
        strcat(coord, ",");
        strcat(coord, yy);
        std::cout << 3 << std::endl ;



    }  
    return src;
}
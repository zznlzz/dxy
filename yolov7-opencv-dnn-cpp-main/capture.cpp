#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "HoughCircles.hpp"
#include "EllipseFitting.hpp"
using namespace std;
using namespace cv;

int main(int argc,char** argv)
{
    //打开摄像头

    VideoCapture cap;//声明相机捕获对象
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    cap.set(CAP_PROP_FRAME_WIDTH,640);//图像的宽
    cap.set(CAP_PROP_FRAME_HEIGHT,480);//图像的高

    int deviceID = 0;//相机设备号
    // http://admin:admin@192.168.1.105:8081
    cap.open("http://admin:admin@192.168.43.1:8081");
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
            imshow("camera",img);
        }else{
            cerr<<"Error: unable to read imagine." << endl;
        }
        
       if (waitKey(10) >= 0) break;
    }
    cap.release();
    destroyAllWindows();

    return 0;
}
#include "yolo.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

#define USE_CUDA true //use opencv-cuda

using namespace std;
using namespace cv;
using namespace dnn;


int main()
{
	string img_path = "/home/zyt/1dxy/yolov7-opencv-dnn-cpp-main/bulid/images/79.jpg";

#if(defined YOLOV5 && YOLOV5==true)
	string model_path = "models/yolov5s.onnx";
#else
	string model_path = "/home/zyt/1dxy/yolov7-opencv-dnn-cpp-main/models/best.onnx";
#endif


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


	VideoCapture cap;//声明相机捕获对象
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
    cap.set(CAP_PROP_FRAME_WIDTH,640);//图像的宽
    cap.set(CAP_PROP_FRAME_HEIGHT,640);//图像的高

    int deviceID = 0;//相机设备号
    // http://admin:admin@192.168.1.105:8081
	//"http://admin:admin@192.168.43.1:8081"
    cap.open(0);
    if (!cap.isOpened())
    {
        cerr << "Error: unable to open camera." << endl;
        return -1;
    }else{
        cout <<"Successfully turned on the camera."<< endl;
    }


	vector<Output> result;
	// Mat img = imread(img_path);
	Mat img;
    while (true)
    {
        cap >> img;
        namedWindow("camera",1);
        if (img.empty() == false)
        {
			Mat out = img;
			// cout << "0" << endl;
            if (test.Detect(img, net, result)) {
				out = test.drawPred(img, result, color);
			}else {
				cout << "Error: detect failed!" << endl;
			}
			// cout << "1" << endl;
			imshow("camera", out);
        }else{
            cerr << "Error: unable to read imagine." << endl;
        }
        
       if (waitKey(10) >= 0) break;
    }
    cap.release();
    cv::destroyAllWindows();
	//system("pause");
	return 0;
}

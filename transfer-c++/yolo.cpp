#include "yolo.h"
#include "global.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;


bool Yolo::readModel(Net &net, string &netPath, bool isCuda = false)
{
	try
	{
		net = readNet(netPath);
	}
	catch (const std::exception &)
	{
		return false;
	}
	// cuda
	if (isCuda)
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	// cpu
	else
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}
#if (defined YOLOV5 && YOLOV5 == false) // yolov7
bool Yolo::Detect(Mat &SrcImg, Net &net, vector<Output> &output, , int model_flag)
{
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = SrcImg.clone();
	if (maxLen > 1.2 * col || maxLen > 1.2 * row)
	{
		Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
		SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
		netInputImg = resizeImg;
	}
	vector<Ptr<Layer>> layer;
	vector<string> layer_names;
	layer_names = net.getLayerNames();
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);
	// 如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	// blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117, 123), true, false);
	// blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(114, 114,114), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR == 6 || CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR == 7
	std::sort(netOutputImg.begin(), netOutputImg.end(), [](Mat &A, Mat &B)
			  { return A.size[2] > B.size[2]; }); // opencv 4.6
#endif
	std::vector<int> classIds;		// 结果id数组
	std::vector<float> confidences; // 结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;	// 每个id矩形框
	float ratio_h = (float)netInputImg.rows / netHeight;
	float ratio_w = (float)netInputImg.cols / netWidth;
	int net_width = className[model_flag].size() + 5; // 输出的网络宽度是类别数+5
	for (int stride = 0; stride < strideSize; stride++)
	{ // stride
		float *pdata = (float *)netOutputImg[stride].data;
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++)
		{ // anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++)
			{
				for (int j = 0; j < grid_x; j++)
				{
					float box_score = sigmoid_x(pdata[4]);
					; // 获取每一行的box框中含有某个物体的概率
					if (box_score >= boxThreshold)
					{
						cv::Mat scores(1, className[model_flag].size(), CV_32FC1, pdata + 5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = sigmoid_x(max_class_socre);
						if (max_class_socre >= classThreshold)
						{
							float x = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * netStride[stride]; // x
							float y = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * netStride[stride]; // y
							float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;			  // w
							float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;			  // h
							int left = (int)(x - 0.5 * w) * ratio_w + 0.5;
							int top = (int)(y - 0.5 * h) * ratio_h + 0.5;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre * box_score);
							boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
						}
					}
					pdata += net_width; // 下一行
				}
			}
		}
	}

	// 执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, nmsScoreThreshold, nmsThreshold, nms_result);
	// 预测框尺寸   预测中的置信度得分   置信度   nms
	output.clear();
	for (int i = 0; i < nms_result.size(); i++)
	{
		int idx = nms_result[i];
		Output result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		// output.clear();
		output.push_back(result);
	}
	if (output.size())
		return true;
	else
		return false;
}
#else
// yolov5
bool Yolo::Detect(Mat &SrcImg, Net &net, vector<Output> &output, int model_flag)
{
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = SrcImg.clone();
	if (maxLen > 1.2 * col || maxLen > 1.2 * row)
	{
		Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
		SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
		netInputImg = resizeImg;
	}
	vector<Ptr<Layer>> layer;
	vector<string> layer_names;
	layer_names = net.getLayerNames();
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);
	// 如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
	// blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117, 123), true, false);
	// blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(114, 114,114), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	std::vector<int> classIds;		// 结果id数组
	std::vector<float> confidences; // 结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;	// 每个id矩形框
	float ratio_h = (float)netInputImg.rows / netHeight;
	float ratio_w = (float)netInputImg.cols / netWidth;
	int net_width = className[model_flag].size() + 5; // 输出的网络宽度是类别数+5
	float *pdata = (float *)netOutputImg[0].data;
	for (int stride = 0; stride < strideSize; stride++)
	{ // stride

		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++)
		{ // anchors
			// const float anchor_w = netAnchors[stride][anchor * 2];
			// const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++)
			{
				for (int j = 0; j < grid_x; j++)
				{
					float box_score = pdata[4];
					; // 获取每一行的box框中含有某个物体的概率
					if (box_score >= boxThreshold)
					{
						cv::Mat scores(1, className[model_flag].size(), CV_32FC1, pdata + 5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = max_class_socre;
						if (max_class_socre >= classThreshold)
						{
							float x = pdata[0]; // (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * m_Mark_Stride[stride];  //x
							float y = pdata[1]; //(sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * m_Mark_Stride[stride];   //y
							float w = pdata[2]; // powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w*ratio_c;   //w
							float h = pdata[3]; // powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h*ratio_r;  //h
							int left = round((x - 0.5 * w) * ratio_w);
							int top = round((y - 0.5 * h) * ratio_h);
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre * box_score);
							boxes.push_back(Rect(left, top, round(w * ratio_w), round(h * ratio_h)));
						}
					}
					pdata += net_width; // 下一行
				}
			}
		}
	}

	// 执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, nmsScoreThreshold, nmsThreshold, nms_result);
	// 预测框尺寸   预测中的置信度得分   置信度   nms
	output.clear();
	for (int i = 0; i < nms_result.size(); i++)
	{
		int idx = nms_result[i];
		Output result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}
	if (output.size())
		return true;
	else
		return false;
}
#endif

Mat Yolo::drawPred(Mat src, vector<Output> result, vector<Scalar> color, int model_flag)
{
	int left_max = 0, right_max = 0, up_max = 0, down_max = 0;
	int left[10], top[10];
	for (int i = 0; i < result.size(); i++)
	{
		
		left[i] = result[i].box.x;
		top[i] = result[i].box.y;
		int color_num = i;
		rectangle(src, result[i].box, color[result[i].id], 2, 5);

		string label = className[model_flag][result[i].id] + ":" + to_string(result[i].confidence);

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine); // 绘制置信度
		top[i] = max(top[i], labelSize.height);
		putText(src, label, Point(left[i], top[i]), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);

		int center_x = left[i] + (result[i].box.width / 2);
		int center_y = top[i] + (result[i].box.height / 2);

		if((left[left_max] + (result[left_max].box.width / 2)) > center_x)
			left_max = i;
		if((left[right_max] + (result[right_max].box.width / 2)) < center_x)
			right_max = i;
		if((top[up_max] + (result[up_max].box.height / 2)) > center_y)
			up_max = i;
		if((top[down_max] + (result[down_max].box.height / 2)) < center_y)
			down_max = i;

		
	}

	int left_x = left[left_max] + result[left_max].box.width/2, left_y = top[left_max] + result[left_max].box.height/2;
	int right_x = left[right_max] + result[right_max].box.width/2, right_y = top[right_max] + result[right_max].box.height/2;
	int up_x = left[up_max] + result[up_max].box.width/2, up_y = top[up_max] + result[up_max].box.height/2;
	int down_x = left[down_max] + result[down_max].box.width/2, down_y = top[down_max] + result[down_max].box.height/2;

	// cv::circle(src, cv::Point(left_x,left_y), 2, cv::Scalar(255,0,0), 3, cv::LINE_AA);// 蓝色
	// cv::circle(src, cv::Point(right_x,right_y), 2, cv::Scalar(0,255,0), 3, cv::LINE_AA);// 绿色
	// cv::circle(src, cv::Point(up_x,up_y), 2, cv::Scalar(0,0,255), 3, cv::LINE_AA);// 红色
	// cv::circle(src, cv::Point(down_x,down_y), 2, cv::Scalar(0,255,255), 3, cv::LINE_AA);// 黄色

	sprintf(coord, "%d,%d,%d", right_x, right_y, flag_servo);
	return src;
}

void Yolo::target(Mat src, vector<Output> result, int model_flag)
{
	int left[10], top[10];
	Mat dst[result.size()];
	for (int i = 0; i < result.size(); i++)
	{
		
		left[i] = result[i].box.x;
		top[i] = result[i].box.y;
		int color_num = i;

		string label = className[model_flag][result[i].id] + ":" + to_string(result[i].confidence);

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine); // 绘制置信度
		top[i] = max(top[i], labelSize.height);

		int center_x = left[i] + (result[i].box.width / 2);
		int center_y = top[i] + (result[i].box.height / 2);

		int x, y, w, h;
		if (result[i].box.width>=result[i].box.height){
			x = left[i];
			y = top[i]-(result[i].box.width-result[i].box.height)/2;
			w = h = result[i].box.width;
		}else{
			x = left[i] - (result[i].box.height-result[i].box.width)/2;
			y = top[i];
			w = h = result[i].box.height;
		}
		
		if (x > src.cols) x = src.cols;
		if (x < 0) x = 0;
		if (y > src.rows) y = src.rows;
		if (y < 0) y = 0;
		if (x+w > src.cols) w = src.cols-x;
		if (y+h > src.rows) h = src.rows-y;
		dst[i] = src(cv::Rect(x, y, w, h));
		resize(dst[i], dst[i], Size(300, 300), 0, 0, INTER_LINEAR);

		string window = "target" + to_string(i);
		imshow(window, dst[i]);
		moveWindow(window, 70+300*i, 0);
	}
	// return dst[0];
}
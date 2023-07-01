# dxy

## **环境** <br>
+ ubuntu22.04 <br>
+ 显卡驱动530.41.03 <br>
+ cuda13.1 <br>
+ cudnn8.9 <br>
+ OpenCV4.6.0 or 4.7.0<br>

### **Circle_detect** <br>
+ 主要使用霍夫圆检测来检测白色圆筒 <br>

### **yolov7-opencv-dnn-cpp-main** <br>
+ 用于将yolo训练好的模型用c++部署  <br>
> models：存放onnx文件（由pt文件经yolo自带的export.py转换出）<br>

### **transfer-c++** <br>
+ 用于将地面站与树莓派通信，传递坐标  <br>
+ 已经将检测圆和H的代码部署进去了，最终是使用这个代码  <br>
+ 下载链接
<a href="transfer-c++/models/bestv5.onnx" target="_blank">bestv5.onnx</a>
<a href="transfer-c++/models/best_circle.onnx" target="_blank">best_circle.onnx</a>

---
+ **6.29更新**
1. 检测H和检测圆筒都替换为使用yolov5
2. 检测圆的传统视觉方案仍保留，增加了筛选白色
3. 使yolo检测坐标也能成功传输
---
+ **6.30更新**
1. 弃用eigen库，改用OpenCV的undistort函数去除图像畸变再进行检测
---
+ **7.1更新**
1. 增加全局变量flag_servo用于控制舵机，增加目标点
+ 传输的coord格式变为"x,y,flag_servo"

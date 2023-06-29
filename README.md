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

---
+ 6.29更新：
1. 检测H和检测圆筒都替换为使用yolov5
2. 检测圆的传统视觉方案仍保留，增加了筛选白色
3. 使yolo检测坐标也能成功传输

# dxy

## **环境** <br>
+ ubuntu22.04 <br>
+ OpenCV4.6.0 <br>


### **Circle_detect** <br>
+ 主要使用霍夫圆检测来检测白色圆筒 <br>

### **data** <br>
+ 用于检测H的数据集及一些数据增强、数据标签转换代码  <br>
> data_enhancement：数据增强、标签转换代码 <br>
> data_yolo：存放原始yolo数据images和labels <br>
> data_xml：存放yolo标签转换后的xml标签 <br>
> data_xml_after:存放xml标签数据增强后的images和labels <br>

### **yolov7-opencv-dnn-cpp-main** <br>
+ 用于将yolo训练好的模型用c++部署  <br>
> models：存放onnx文件（由pt文件经yolo自带的export.py转换出）<br>

### **transfer-c++** <br>
+ 用于将地面站与树莓派通信，传递坐标  <br>
+ 已经将检测圆和H的代码部署进去了，最终是使用这个代码  <br>

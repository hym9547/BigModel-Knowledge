# YOLOV5
## 数据集
- datasets/name 的目录方式
- 在name下设置train、valid、test三个目录，其下设置 images和labels两个文件夹，分别存放image和label，名称对应
- label 为 target xmin ymin xmax ymax 形式的txt文件（空格相隔，换行显示新的检测框）

## 类别
-  ```data`` 下的yaml文件，直接复制一个，修改路径和names即可

## 模型
- ```models``` 下存在不同的类型（n、s、l、m、x）->（速度与精度相反：n速度最快，精度最低）
- 修改模型层次：对应型号文件下的模型参数修改

## 训练
- 命令 ：python train.py --img 640 --batch 4 --epochs 50 --data data/Data.yaml --weights yolov5n.pt --device 0
  - img -> 图片resize 
  - batch -> 每次训练的图片数量
  - epochs -> 总轮数
  - data -> 数据集配置文件
  - cfg -> 指定训练配置文件 默认yolov5n.yaml
  - weights -> 预训练权重
  - device -> 驱动 数字表示gpu

## 测试
- ```torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)``` 加载模型
- ```model(img_source)``` 对图片进行检测
- ```results.render()[0]``` 带标注的图片
- ```results.pandas().xyxy[0].to_dict('records')``` 检测结果数据

# YOLOV8//11
## 数据集
- datasets/name 的目录方式
- 在name下设置train、valid、test三个目录，其下设置 images和labels两个文件夹，分别存放image和label，名称对应
- label 为 target xmin ymin xmax ymax 形式的txt文件（空格相隔，换行显示新的检测框）
## 类别
- cfg/datasets/Data.yaml 设置数据集路径\检测目标类别
## 模型
- cfg/models/下存在不同的版本号（v8\v5\11等），选择后目录后内容为多个任务目标
  - 正矩形框检测，无后缀
  - 斜矩形框检测，后缀为 obb
  - 分类,后缀为 cls
  - 姿势检测,后缀为 pose
  - 语义分割，后缀为 seg
- 进入yaml文件后，设置nc（类别总数量）
- 若需要调整模型层次或组件，调整此次yaml文件的模型层信息，例如：（在目标检测中新增一层混合注意力）
  - cfg/models/11/yolo11.yaml 的backbone添加  - [-1, 1, CBAM, [1024, 7]] # 9 ,后面所有层次+1
  - nn/modules/ 选择一个同类文件，本次添加混合注意力，同卷积，所以在conv.py文件操作
  - conv.py 添加 class CBAM；在__all__ 中添加导出
  - __init__.py 添加 导入和导出
  - tasks.py 中添加导入；base_modules中添加引用和解析

## 训练
- 命令 ：python detect train imgsz=640 batch=4 epochs=50 data=data/Data.yaml model=cfg/models/11/yolo11n.yaml device=0
  - imgsz -> 图片resize 
  - batch -> 每次训练的图片数量
  - epochs -> 总轮数
  - data -> 数据集配置文件
  - model -> 指定模型配置文件，文件为yolo11.yaml n表示型号，也可以直接写yolo11m.yaml，表示从yolo11.yaml中使用m型号
  - device -> 驱动 数字表示gpu
```python
from ultralytics import YOLO
model = YOLO.model('cfg/models/11/yolov11n.yaml').load('yolo11n.pt')
result = model.train(data = 'cfg/datasets/Data.yaml',
                     epoches=500,
                     imgze = 1920,
                     batch = 8,
                     patient = 50,
                     option = 'Adawm',
                     lr0 = 0.001
)
```
## 测试
- python detect.py weights=best.pt source=1.jpg
```python
model = YOLO.model(model_path) # 加载模型
results = model(img_source)# 对图片进行检测
for result in results:
  result.cls  # 分类
  result.xyxy  # 坐标
  result.show()   # 展示
```
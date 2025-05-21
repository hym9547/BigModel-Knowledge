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
# Conda

| 对比 |         Anaconda          | Miniconda | Anaconda Server | Conda-Forge |
| :--: | :-----------------------: | :-------: | :-------------: | :---------: |
| 定位 |           完整            |   轻量    |     企业级      |    社区     |
| 场景 | 初学者/空间足够（2G至少） | 灵活控制  |    企业机构     | 试用最新库  |

## 安装 Anaconda

- **```下载地址```**：[Download Now | Anaconda](https://www.anaconda.com/download/success)；一直点next（安装位置修改为空间足够那个位置，非C）

- **```环境变量```**：用户/系统 的**path**添加 Anaconda 安装目录、Scripts、Library\bin
  - D:\anconda1 
  - D:\anconda1\Scripts 
  - D:\anconda1\Library\bin

- **```验证```**：
  - cmd: conda --version / conda -v （一般都是--全程 等价于 -首字母 ；例如 --name == -n）
  - 开始菜单搜索 **Anaconda prompt**
- **```修改虚拟环境安装位置```**：
  - **conda info**：找到**condarc **文件地址、**base environment** 地址 、 **conda av data dir **地址、**envs directories**地址
  - 进入**condarc **文件，修改**envs_dirs**、**pkgs_dirs**
  - 返回查看**base environment** 地址 、 **conda av data dir **地址、**envs directories**地址的变化
  - 若存在安装环境后，还是存在默认路径（C盘），修改用户权限 或者 在创建时 添加指令 **--prefix=/home/conda_env/mmcv**，用于指定安装位置

## 创建虚拟环境

- **```创建指令```**：conda create -n test python=3.10 ；如此便创建了一个pyhton3.10版本的虚拟环境（conda 指令 使用 =，在pip中使用 ==）
  - 进入**```test```**环境：**activate test **
- **```安装包```**： 
  - **pip install  numpy / pip install numpy==3.25**
  - **conda install  numpy / conda install numpy=3.25**
  - 指定版本不存在时，自动输出存在的版本，从中挑选一个相近的即可

## 删除虚拟环境

- **conda env remove -n test**

## 配置镜像源

- **作用**：加速下载某些包
- **指令**：conda config --add url
  - conda config --add https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

 ## Anaconda常用命令

- **conda list**：查看环境中的所有包，等价 **pip list**
- **conda install package**：安装包 等价 **pip install**
- **conda remove package**：删除包 等价 **pip uninstall**
- **conda env list**：列出所有环境
- **activate envname（或 source activate envname）**：启用/激活环境
- **deactivate（或 source deactivate）**：退出环境
- **conda clean --all**：清理所有缓存（文件、包、临时文件） 
- **conda clean --packages**：清理未使用的缓存包
- **conda clean --tarballs**：清理压缩包
- **conda clean --index-cache**：清理索引缓存
- **--force**：强制的意思，可以加载其他命令里面
- **pip cache purge**：清理使用pip命令的缓存
# Orbbec相机连接和二次开发
## 1. OrbbecView 软节调试
![alt text](../image/OrbbecViewer.png)
### 重点官方要求安装 Linux udev 规则的
```sh
cd ~/OrbbecViewer_v1.10.27_202509260133_linux_x64_release/scripts
sudo chmod +x install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```
运行命令后，要重新插拔相机的数据线，就可以成功连接了。
软节详细用法介绍可以参考官方的文件：
https://github.com/orbbec/OrbbecSDK/blob/main/doc/OrbbecViewer/English/OrbbecViewer.md
## 2. pyorbbecsdk

首次下载源码：
```sh
git clone -b main \
https://github.com/orbbec/pyorbbecsdk.git
```
首次下载要按照官方步骤重新编译：
```sh
cd ~/PiPER_X/orbbec/pyorbbecsdk

rm -rf build install

python3 -m pip install -r requirements.txt

mkdir build
cd build
cmake -Dpybind11_DIR="$(pybind11-config --cmakedir)" ..
make -j$(nproc)
make install
cd ~/PiPER_X/orbbec/pyorbbecsdk
export PYTHONPATH="$(pwd)/install/lib:$PYTHONPATH"

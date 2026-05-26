MindSpeed-Bridge 的安装指导：

## 1. 依赖配套总览

MindSpeed-Bridge 的依赖配套如下表：

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">25.5.0</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>CANN Toolkit（开发套件）</td>
      <td rowspan="3">CANN 9.0.0</td>
  </tr>
  <tr>
    <td>CANN Ops（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.11</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>2.9.0</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
    <td >7.3.0</td>
  </tr>
  <tr>
    <td>MindSpeed</td>
    <td >core_r0.16.0</td>
  </tr>
</table>

## 2. 依赖安装

### 2.1 驱动固件安装

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.5.0&driver=Ascend+HDK+25.5.0)，请根据系统和硬件产品型号选择对应版本的 `driver` 和 `firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=Ubuntu)或执行以下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --force
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

### 2.2 CANN 安装

下载 [CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据系统选择 `aarch64` 或 `x86_64` 对应版本的 `cann-toolkit`、`cann-ops` 和 `cann-nnal`。参考 [CANN 安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=Ubuntu) 或执行以下命令安装：

```shell
# 因为版本迭代，包名存在出入，根据实际修改
chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
./Ascend-cann-toolkit_<version>_linux-<arch>.run --install
chmod +x Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run
./Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run --install
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
chmod +x Ascend-cann-nnal_<version>_linux-<arch>.run
./Ascend-cann-nnal_<version>_linux-<arch>.run --install
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh # 修改为实际安装的nnal包路径
```

### 2.3 PyTorch 安装

准备 [Torch_npu](https://www.hiascend.com/developer/download/community/result?module=pt)，执行以下命令安装或参考 [Ascend Extension for PyTorch 配置与安装](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)：

```shell
# 安装torch和torch_npu 构建参考 https://gitcode.com/ascend/pytorch/releases
pip3 install torch-2.9.0-cp311-cp311-manylinux_2_28_aarch64.whl
pip3 install torch_npu-2.9.0-cp311-cp311-manylinux_2_28_aarch64.whl
```

### 2.4 仓库依赖安装

```shell
# 使能环境变量
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh # 修改为实际安装的nnal包路径

# 准备Megatron-LM源码
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/
git checkout core_v0.16.1
pip install -e .
cd ..

# 安装MindSpeed加速库
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed/
git checkout core_r0.16.0
pip install -r requirements.txt
pip install -e .
cd ..

# 安装MindSpeed-Bridge
git clone https://gitcode.com/ascend/MindSpeed-Bridge.git
cd MindSpeed-Bridge/
pip install -r requirements.txt
cd ..

# Megatron-Bridge源码下载
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge/
git checkout v0.3.1
cd ..
```

## 3. Bridge 适配迁移

### 3.1：自动适配（推荐）

执行以下命令，自动完成依赖仓库下载与适配迁移：

```shell
cd MindSpeed-Bridge/
bash tools/install_auto.sh --download
```

> 若依赖仓库已手动准备好，可省略 `--download` 参数，仅执行适配迁移：
> ```shell
> bash tools/install_auto.sh
> ```

### 3.2：手动适配

-  **复制工作目录**

将本仓库中的 `mindspeed_bridge` 文件夹，复制到 **Megatron-Bridge** 根目录下：

```shell
cd Megatron-Bridge/
cp -r ../MindSpeed-Bridge/mindspeed_bridge ./
```

- **注册模型**

修改文件：`src/megatron/bridge/models/__init__.py`

原文件193行加入：

```python
from mindspeed_bridge.models import *
```

- **注册配置**

修改文件：`src/megatron/bridge/recipes/__init__.py`

原文件35行加入：

```python
from mindspeed_bridge.recipes import *
```

-  **适配训练入口**

修改文件：`scripts/training/run_recipe.py`

原文件62行加入：

```python
import mindspeed.megatron_adaptor
```

原文件72行加入：

```python
from mindspeed_bridge.models.qwen_vl import qwen3_vl_forward_step
```

原文件77行注册前向函数：

```python
STEP_FUNCTIONS: dict[str, Callable] = {
    "gpt_step": gpt_forward_step,
    "vlm_step": vlm_forward_step,
    "llava_step": llava_forward_step,
    "qwen3_vl_step": qwen3_vl_forward_step,
}
```

原文件237行注册mindspeed的patch参数：

```python
from mindspeed.megatron_adaptor import repatch
from dataclasses import asdict
repatch(asdict(config.model))
```

> [!NOTE]
>
> Megatron-Bridge 0.3.1 暂未原生支持 qwen35_vl，必须手动注册该前向函数


## 4. Bridge启动训练

修改 `mindspeed_bridge/examples/` 路径下脚本启动训练：

```shell
bash mindspeed_bridge/examples/models/vlm/qwen35_vl/qwen35_vl_35b_sft.sh
```

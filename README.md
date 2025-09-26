---
pipeline_tag: text-to-video
frameworks:
  - PyTorch
hardwares:
  - NPU
  - Atlas 800T A2
  - Atlas 800I A2
license: apache-2.0
---
# Wan2.2推理指导
## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.11.10 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I/800T A2(8*64G)推理设备：支持的卡数最小为1
[Atlas 800I/800T A2(8*64G)](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
chmod +x ./Ascend-cann-nnal_{version}_linux-{arch}.run  (若使用稀疏FA)
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
./Ascend-cann-nnal{version}_linux-{arch}.run --check  (若使用稀疏FA)
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install
./Ascend-cann-nnal{version}_linux-{arch}.run --torch_atb --install  (若使用稀疏FA)

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 1.3 环境依赖安装
```shell
pip3 install -r requirements.txt
```

### 1.4 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.5 Torch_npu安装
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

### 1.6 gcc、g++安装
# 若环境镜像中没有gcc、g++，请用户自行安装
yum install gcc
yum install g++

# 导入头文件路径
export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH

注：若使用openeuler镜像，需要配置gcc、g++环境，否则会导致`fatal error: 'stdio.h' file not found`

## 二、下载权重

### Wan2.2 权重及配置文件说明

-  Huggingface

|  模型 | 链接  |
| ------------ | ------------ |
| Wan2.2-T2V-A14B  |  [🤗huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B/tree/main) | 
| Wan2.2-I2V-A14B  |  [🤗huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/tree/main) | 
| Wan2.2-TI2V-5B  |  [🤗huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/main) |   


- Modelers

|  模型 | 链接  |
| ------------ | ------------ |
| Wan2.2-T2V-A14B  |  [	Modelers](https://modelers.cn/models/Modelers_Park/Wan2.2-T2V-A14B) | 
| Wan2.2-I2V-A14B  |  [	Modelers](https://modelers.cn/models/Modelers_Park/Wan2.2-I2V-A14B ) |  
| Wan2.2-TI2V-5B  |  [ Modelers](https://modelers.cn/models/Modelers_Park/Wan2.2-TI2V-5B) |  


## 三、Wan2.2使用

### 3.1 下载到本地
```shell
git clone https://modelers.cn/MindIE/Wan2.2.git
```

### 3.2 Wan2.2-T2V-A14B
使用上一步下载的权重
```shell
model_base="./Wan2.2-T2V-A14B/"
```
#### 3.2.1 等价优化
#### 3.2.1.1 8卡性能测试
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 --master_port=23459 generate.py \
--task t2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--vae_parallel \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--base_seed 0

```
参数说明：
- ALGO: 为0表示默认FA算子; 设置为1表示使用高性能FA算子
- task: 任务类型。
- ckpt_dir: 模型的权重路径
- size: 生成视频的分辨率，支持(1280,720)、(832,480)分辨率
- frame_num: 生成视频的帧数
- sample_steps: 推理步数
- dit_fsdp: dit使能fsdp, 用以降低显存占用
- t5_fsdp: t5使能fsdp, 用以降低显存占用
- cfg_size: cfg并行数
- ulysses_size: ulysses并行数
- vae_parallel: 使能vae并行策略
- prompt: 文本提示词
- base_seed: 随机种子

#### 3.2.1.2 16卡性能测试
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=16 --master_port=23459 generate.py \
--task t2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 2 \
--ulysses_size 8 \
--vae_parallel \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--base_seed 0
```

#### 3.2.2 算法优化-稀疏FA
#### 3.2.2.1 8卡性能测试
执行命令：
```shell
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 --master_port=23459 generate.py \
--task t2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--vae_parallel \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--use_rainfusion \
--sparsity 0.64 \
--sparse_start_step 15 \
--base_seed 0
```
参数说明：
- use_rainfusion: 使能稀疏flash attention计算
- sparsity: 稀疏度，值为[0, 1), 稀疏度越大，加速比越高，相应精度损失更大
- spasre_start_step: 开始稀疏的时间步，通常需要保证不小于1/4的总时间步数


### 3.3 Wan2.2-I2V-A14B
使用上一步下载的权重
```shell
model_base="./Wan2.2-I2V-A14B/"
```

#### 3.3.1 等价优化
#### 3.3.1.1 8卡性能测试

执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 generate.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--vae_parallel \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0
```
参数说明：
- ALGO: 为0表示默认FA算子; 设置为1表示使用高性能FA算子
- task: 任务类型。
- ckpt_dir: 模型的权重路径
- size: 生成视频的分辨率，支持(1280,720)、(832,480)分辨率
- frame_num: 生成视频的帧数
- sample_steps: 推理步数
- dit_fsdp: dit使能fsdp, 用以降低显存占用
- t5_fsdp: t5使能fsdp, 用以降低显存占用
- cfg_size: cfg并行数
- ulysses_size: ulysses并行数
- vae_parallel: 使能vae并行策略
- image: 输入图片路径
- prompt: 文本提示词
- base_seed: 随机种子

#### 3.3.1.2 16卡性能测试
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=16 --master_port=23459 generate.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 2 \
--ulysses_size 8 \
--vae_parallel \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0
```

#### 3.3.2 等价优化--稀疏FA
#### 3.3.2.1 8卡性能测试
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 generate.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--vae_parallel \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--use_rainfusion \
--sparsity 0.64 \
--sparse_start_step 15 \
--base_seed 0
```

参数说明：
- use_rainfusion: 使能稀疏flash attention计算
- sparsity: 稀疏度，值为[0, 1), 稀疏度越大，加速比越高，相应精度损失更大
- spasre_start_step: 开始稀疏的时间步，通常需要保证不小于1/4的总时间步数


### 3.4 Wan2.2-TI2V-5B
使用上一步下载的权重
```shell
model_base="./Wan2.2-TI2V-5B/"
```

#### 3.4.1 等价优化
#### 3.4.1.1 单卡性能测试
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

python generate.py \
--task ti2v-5B \
--ckpt_dir ${model_base} \
--size 1280*704 \
--frame_num 121 \
--sample_steps 50 \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0 
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- task: 任务类型。
- ckpt_dir: 模型的权重路径
- size: 生成视频的分辨率，支持(1280,720)、(832,480)分辨率
- frame_num: 生成视频的帧数
- sample_steps: 推理步数
- image: 输入图片路径
- prompt: 文本提示词
- base_seed: 随机种子


#### 3.4.1.2 8卡性能测试

执行命令:
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
torchrun --nproc_per_node=8 generate.py \
--task ti2v-5B \
--ckpt_dir ${model_base} \
--size 1280*704 \
--frame_num 121 \
--sample_steps 50 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- task: 任务类型。
- ckpt_dir: 模型的权重路径
- size: 生成视频的分辨率，支持(1280,720)、(832,480)分辨率
- frame_num: 生成视频的帧数
- sample_steps: 推理步数
- dit_fsdp: dit使能fsdp, 用以降低显存占用
- t5_fsdp: t5使能fsdp, 用以降低显存占用
- cfg_size: cfg并行数
- ulysses_size: ulysses并行数
- vae_parallel: 使能vae并行策略
- image: 输入图片路径
- prompt: 文本提示词
- base_seed: 随机种子

#### 3.4.1.3 16卡性能测试
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=16 --master_port=23459 generate.py \
--task ti2v-5B \
--ckpt_dir ${model_base} \
--size 1280*704 \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 2 \
--ulysses_size 8 \
--vae_parallel \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0
```

#### 3.4.2 算法优化
#### 3.4.2.1 单卡性能测试
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

python generate.py \
--task ti2v-5B \
--ckpt_dir ${model_base} \
--size 1280*704 \
--frame_num 121 \
--sample_steps 50 \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0 \
--use_attentioncache \
--start_step 20 \
--attentioncache_interval 2 \
--end_step 47
```
参数说明：
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- use_attentioncache: 使能attentioncache策略
- start_step: cache开始的step
- attentioncache_interval: cache重计算间隔
- end_step: cache结束的step

#### 3.4.2.2 8卡性能测试

执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
torchrun --nproc_per_node=8 generate.py \
--task ti2v-5B \
--ckpt_dir ${model_base} \
--size 1280*704 \
--frame_num 121 \
--sample_steps 50 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--vae_parallel \
--base_seed 0 \
--use_attentioncache \
--start_step 20 \
--attentioncache_interval 2 \
--end_step 47
```

#### 3.4.2.1 16卡性能测试
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=16 --master_port=23459 generate.py \
--task ti2v-5B \
--ckpt_dir ${model_base} \
--size 1280*704 \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 2 \
--ulysses_size 8 \
--vae_parallel \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--vae_parallel \
--base_seed 0 \
--use_attentioncache \
--start_step 20 \
--attentioncache_interval 2 \
--end_step 47
```


## 四、量化功能支持
本项目新增量化功能，支持权重 8 位（w8）与激活 8 位（a8）的量化组合，可减少模型显存占用并保持推理性能
### 4.1 安装量化工具msModelSlim
参考[官方README](https://gitee.com/ascend/msit/tree/master/msmodelslim)
1. git clone下载msit仓代码
2. 进入到msit/msmodelslim的目录 cd msit/msmodelslim；并在进入的msmodelslim目录下，运行安装脚本 bash install.sh

### 4.2 量化模型生成
要获取对量化模型的描述文件和权重文件，需在原有generate.py命令中添加`--quant_mode`参数（设置值为2），添加`--quant_data_dir`参数（指向量化描述文件和量化权重存放的文件夹路径），其余参数与原生模型推理一致。

#### 4.2.1 量化参数说明
| 参数 | 含义 | 可选值 | 默认值 |
|------|------|--------|--------|
| --quant_mode | 量化模式 | 0: 不使用量化模型推理  1: 导出校准数据， 2：导出量化模型， 3：使用量化模型推理. | 0 |
| --quant_data_dir | 量化模型保存目录 | - | ./output/quant_data |

#### 4.2.2 Wan2.2-T2V-A14B
执行命令：

```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export model_path="your local Wan2.2-T2V-A14B model path"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=23459 generate.py \
  --task t2v-A14B \
  --ckpt_dir ${model_path} \
  --size 1280*720 \
  --frame_num 81 \
  --sample_steps 40 \
  --t5_fsdp \
  --cfg_size 1 \
  --ulysses_size 8 \
  --vae_parallel \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 0 \
  --offload_model False \
  --dit_fsdp \
  --quant_mode 2 \
  --quant_data_dir "./output/quant_data"
```

#### 4.2.3 Wan2.2-I2V-A14B
执行命令：

```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export model_path="your local Wan2.2-I2V-A14B model path"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=23459 generate.py \
  --task i2v-A14B \
  --ckpt_dir ${model_path} \
  --size 1280*720 \
  --frame_num 81 \
  --sample_steps 40 \
  --t5_fsdp \
  --cfg_size 1 \
  --ulysses_size 8 \
  --vae_parallel \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --base_seed 0 \
  --offload_model False \
  --dit_fsdp \
  --quant_mode 2 \
  --quant_data_dir "./output/quant_data"
```

#### 4.2.4 Wan2.2-TI2V-5B
执行命令：

```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export model_path="your local Wan2.2-TI2V-5B model path"

export ASCEND_RT_VISIBLE_DEVICES=2 # 设置为2卡
python generate.py \
  --task ti2v-5B \
  --ckpt_dir ${model_path} \
  --size 1280*704 \
  --frame_num 121 \
  --sample_steps 50 \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --base_seed 0 \
  --use_attentioncache \
  --start_step 20 \
  --attentioncache_interval 2 \
  --end_step 47 \
  --quant_mode 2 \
  --quant_data_dir "./output/quant_data"
```

### 4.3 安装量化模型推理工具NNAL神经网络加速库和torch_atb
#### 4.3.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)

#### 4.3.2 安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x Ascend-cann-nnal_<version>_linux-<arch>.run
# 默认路径安装:
./Ascend-cann-nnal_<version>_linux-<arch>.run --install --torch_atb
# 配置环境变量:
source ${HOME}/Ascend/nnal/atb/set_env.sh
```

### 4.4 使用量化模型推理
使用量化模型进行推理时，需在原有generate.py命令中添加`--quant_mode`参数（设置值为3），添加`--quant_data_dir`参数（指向量化描述文件和量化权重存放的文件夹路径），其余参数与原生模型推理一致。
#### 4.4.1 Wan2.2-T2V-A14B
执行命令：（以等价优化 8卡 为例）

```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export model_path="your local Wan2.2-T2V-A14B model path"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=23459 generate.py \
  --task t2v-A14B \
  --ckpt_dir ${model_path} \
  --size 1280*720 \
  --frame_num 81 \
  --sample_steps 40 \
  --t5_fsdp \
  --cfg_size 1 \
  --ulysses_size 8 \
  --vae_parallel \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 0 \
  --offload_model False \
  --dit_fsdp \
  --quant_mode 3 \
  --quant_data_dir "./output/quant_data"
```

#### 4.4.2 Wan2.2-I2V-A14B
执行命令：（以等价优化 8卡 为例）
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export model_path="your local Wan2.2-I2V-A14B model path"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=23459 generate.py \
  --task i2v-A14B \
  --ckpt_dir ${model_path} \
  --size 1280*720 \
  --frame_num 81 \
  --sample_steps 40 \
  --t5_fsdp \
  --cfg_size 1 \
  --ulysses_size 8 \
  --vae_parallel \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --base_seed 0 \
  --offload_model False \
  --dit_fsdp \
  --quant_mode 3 \
  --quant_data_dir "./output/quant_data"
```

#### 4.4.3 Wan2.2-TI2V-5B
##### 4.4.3.1 等价优化 单卡
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export model_path="your local Wan2.2-TI2V-5B model path"

export ASCEND_RT_VISIBLE_DEVICES=2
python generate.py \
  --task ti2v-5B \
  --ckpt_dir ${model_path} \
  --size 1280*704 \
  --frame_num 121 \
  --sample_steps 50 \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --base_seed 0 \
  --use_attentioncache \
  --start_step 20 \
  --attentioncache_interval 2 \
  --end_step 47 \
  --quant_mode 3 \
  --quant_data_dir "./output/quant_data"
```

##### 4.4.3.2 等价优化 8卡
执行命令：
```shell
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export model_path="your local Wan2.2-TI2V-5B model path"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 generate.py \
  --task ti2v-5B \
  --ckpt_dir ${model_path} \
  --size 1280*704 \
  --frame_num 121 \
  --sample_steps 50 \
  --t5_fsdp \
  --cfg_size 1 \
  --ulysses_size 8 \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --base_seed 0 \
  --offload_model False \
  --dit_fsdp \
  --quant_mode 3 \
  --quant_data_dir "./output/quant_data"
```


## 五、推理结果参考
###  Atlas 800I A2(8*64G) 64核(arm)性能数据 (ALGO=1)

| 模型 | 分辨率 | 帧数 | 迭代次数 | 卡数 | E2E耗时|
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Wan2.2-T2V-A14B | 1280×720 | 81 | 40 | 8 | 435.99s |
| Wan2.2-I2V-A14B | 1280×720 | 81 | 40 | 8 | 436.42s |
| Wan2.2-TI2V-5B | 1280×704 | 121 | 50 | 8 |72.21s |


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。

##六、常见问题
1. 若出现OOM, 可添加环境变量 `export T5_LOAD_CPU=1`，以降低显存占用
2. 当前仅TI2V支持attentioncache
3. 若遇到报错: `Directory operation failed. Reason: Directory [/usr/local/Ascend/mindie/latest/mindie-rt/aoe] does not exist`,请设置环境变量`unset TUNE_BANK_PATH`
4. 若使用openeuler镜像, 若没有配置gcc、g++环境，会遇到报错：`fatal error: 'stdio.h' file not found`，请参考1.6 gcc、g++安装
5、若循环跑纯模型推理，可能会因为HCCL端口未及时释放，导致因端口被占用而推理失败，报错：`Failed to bind the IP port. Reason: The IP address and port have been bound already.`
  `HCCL function error :HcclGetRootInfo(&hcclID), error code is 7`:  请配置`export HCCL_HOST_SOCKET_PORT_RANGE="auto"`不指定端口
  `HCCL function error :HcclGetRootInfo(&hcclID), error code is 11`: 请配置`sysctl -w net.ipv4.ip_local_reserved_ports=60000-60015`预留端口
  
   

# Pixal3D ROCm10 异构双卡适配与 ComfyUI 集成记录

> 适用环境：AMD Radeon AI PRO R9700 32GB + Radeon RX 7900 XTX 24GB  
> ComfyUI：0.37.4  
> PyTorch：2.13.0+rocm10.0.0  
> Pixal3D：ComfyUI 原生节点集成方案  
> 记录时间：2026-09-30

## 一、目标

在现有 ComfyUI ROCm10 环境中实现 Pixal3D 的稳定双卡运行，并保持：

- R9700 作为主模型计算卡；
- 7900 XTX 承担 Shape VAE、Texture VAE；
- Structure / Shape 阶段使用双卡 CFG Split；
- 避免一遇到错误就把所有模型退回单卡；
- 修复 Trellis2 / Pixal3D 原生节点在异构双卡下的 GPU device routing 问题；
- 修复 AMD ROCm 下跨 GPU direct copy 引发的 HSA memory fault；
- 修复 Save3DAdvanced 输出 GLB 在旧版 ComfyUI Frontend 中无法加载皃�{�题。

## 二、硬件与设备映射

### 实际硬件

```text
GPU 0：AMD Radeon AI PRO R9700 32GB
GPU 1：AMD Radeon RX 7900 XTX 24GB
```

ComfyUI / PyTorch 内部映射：

```text
cuda:0 = Radeon AI PRO R9700
cuda:1 = Radeon RX 7900 XTX
```

### 最终职责划分

```text
R9700 / gpu:0
└─ Pixal3D BF16 主模型

7900 XTX / gpu:1
├─ Shape VAE
└─ Texture VAE

R9700 + XTX
├─ Structure CFG Split
└─ Shape CFG Split
```

32GB 主卡主要承担 Pixal3D BF16 主模型和采样，24GB XTX 负责两个 VAE，并参与适合的 CFG 并行阶段。

## 三、模型文件

当前 ComfyUI 使用：

```text
models/diffusion_models/pixal3d_bf16.safetensors
models/vae/trellis_2_shape_vae_bf16.safetensors
models/vae/trellis_2_texture_vae_bf16.safetensors
models/clip_vision/dino_v3_L_naf_fp32.safetensors
models/geometry_estimation/moge_2_vitl_normal_fp16.safetensors
models/background_removal/birefnet.safetensors
```

共享目录：

```text
/app/share/comfyui/models/
```

宿主机目录：

```text
/home/victor/ai_share/comfyui/models/
```

## 四、最终工作流

生产工作流：

```text
/app/share/comfyui/user/default/workflows/Pixal3D_BF16_DualGPU_CFGSplit_R9700_XTX.json
```

### 主模型

```text
UNETLoader
  ↓
SelectModelDevice
device = gpu:0
```

标题：

```text
R9700 · Pixal3D BF16
```

### Shape VAE

```text
VAELoader
  ↓
SelectVAEDevice
device = gpu:1
```

标题：

```text
7900 XTX · Shape VAE
```

### Texture VAE

```text
VAELoader
  ↓
SelectVAEDevice
device = gpu:1
```

标题：

```text
7900 XTX · Texture VAE
```

### CFG 双卡

Structure：

```text
MultiGPU_WorkUnits
max_gpus = 2
```

Shape：

```text
MultiGPU_WorkUnits
max_gpus = 2
```

## 五、官方模板中的 Trellis2 残留问题

官方 Pixal3D/Trellis2 模板保留了 Trellis2 切换支路。即使 Boolean 为 `false`，ComfyUI 运行前仍会校验整个图，因此会报：

```text
trellis_2_int8_convrot.safetensors
```

缺失。

当前生产方案只使用 Pixal3D BF16，因此删除了：

- Trellis2 主模型 Loader
- Trellis2Conditioning
- Boolean 切换开关
- 三个相关 Switch
- Trellis2 / INT8 旧说明

最终只保留纯 Pixal3D BF16 路径。

## 六、第一次双卡崩溃

### 现象

Structure sampling 能完成，但进入 Shape VAE 后出现：

```text
HSA_STATUS_ERROR_MEMORY_FAULT
Memory Fault Error
kernel: direct_copy
```

并伴随：

```text
Fatal Python error: Aborted
Fatal Python error: Segmentation fault
```

ROCm 还会生成巨大 GPU coredump：

```text
/app/share/comfyui/app/gpucore.1.gpu
```

实测一次可达到约 9GB 到 18GB。

### 根因

虽然工作流使用：

```text
SelectVAEDevice -> gpu:1
```

但 ComfyUI 原生 Trellis2 节点内部仍使用：

```python
comfy.model_management.get_torch_device()
```

该函数返回主设备：

```text
cuda:0
```

从而形成：

```text
Shape VAE 权重 -> cuda:1 / XTX
输入张量       -> cuda:0 / R9700
```

随后在异构双卡之间发生错误的 direct copy / P2P 路径，触发 HSA page fault。

## 七、第一阶段修复：尊重 VAE 自己的 device

修改：

```text
/app/share/comfyui/app/comfy_extras/nodes_trellis2.py
```

涉及：

```text
VaeDecodeStructureTrellis2
VaeDecodeShapeTrellis
VaeDecodeTextureTrellis
Trellis2UpsampleStage
```

原逻辑：

```python
device = comfy.model_management.get_torch_device()
```

改为：

```python
device = vae.device
```

Structure 节点对应：

```python
load_device = vae.device
```

这样：

```text
SelectVAEDevice -> gpu:1
```

才真正控制 Trellis2 / Pixal3D VAE 的执行设备。

## 八、第二次崩溃与第二层问题

第一阶段修复后，流程已经能继续完成：

```text
Structure sampling
Shape VAE
20/20
12/12
```

但后续仍发生：

```text
HSA_STATUS_ERROR_MEMORY_FAULT
kernel: __amd_rocclr_copyBuffer
```

说明问题不止是节点选错 device，异构 GPU 直接 tensor copy 本身也可能触发 ROCm fault。

## 九、第二阶段修复：host-staged 跨 GPU tensor transfer

对关键跨卡数据移动改成：

```text
GPU A
 ↓
CPU host memory
 ↓
GPU B
```

避免：

```text
GPU A -> GPU B direct P2P
```

也就是说，只把“跨卡交接”经过 CPU，中间计算仍在目标 GPU 上执行，并不是把模型改成 CPU 推理。

修复覆盖：

```text
Shape VAE samples
Shape VAE coords
Texture VAE samples
Texture VAE coords
Structure VAE input
Upsample VAE input
SparseTensor feats
SparseTensor coords
```

目的：规避 R9700 + 7900 XTX 异构组合下 ROCm direct-copy page fault。

## 十、第三阶段修复：显式 GPU device scope

进一步发现，即使 tensor 已经位于 `cuda:1`，Trellis2 内部某些 sparse/custom op 仍可能依赖：

```python
torch.cuda.current_device()
```

如果当前全局 device 仍是 `cuda:0`，底层算子仍可能选错 GPU。

因此对关键 VAE 调用增加类似：

```python
with torch.cuda.device(device):
    ...
```

确保执行 VAE 时当前设备上下文就是 XTX。

覆盖：

```text
VaeDecodeStructureTrellis2
VaeDecodeShapeTrellis
VaeDecodeTextureTrellis
Trellis2UpsampleStage
```

日志验证：

```text
Trellis2 Structure VAE: selected=cuda:1
Trellis2 Upsample Shape VAE: selected=cuda:1
Trellis2 Shape VAE: selected=cuda:1
Trellis2 Texture VAE: selected=cuda:1
```

并且：

```text
Trellis2 Shape VAE: decode_shape_slat begin on cuda:1
Trellis2 Shape VAE: decode_shape_slat complete

Trellis2 Texture VAE: decode_tex_slat begin on cuda:1
Trellis2 Texture VAE: decode_tex_slat complete
```

这说明 VAE 不只是权重在 XTX，实际 decode 也在 XTX 上执行。

## 十一、双卡占用验证

采样阶段实测：

```text
R9700     100%
7900 XTX  100%
```

说明 `MultiGPU CFG Split` 确实让两张卡同时工作，不是副卡只占显存。

整体结构保持：

```text
R9700 -> Pixal3D diffusion / sampling
XTX   -> Shape / Texture VAE
双卡  -> Structure / Shape CFG Split
```

## 十二、完整运行验证

### 1024

完整通过：

```text
Prompt executed in 174.85 seconds
```

### 1536

完整通过：

````text
Prompt executed in 170.21 seconds
```

并继续完成：

```text
Remesh
QEM simplify
UV unwrap
Bake texture
Save3DAdvanced
```

没有再次出现：

```text
HSA_STATUS_ERROR_MEMORY_FAULT
Memory access fault
Segmentation fault
```

测试结束：

```text
queue_running = 0
queue_pending = 0
gpucore = []
```

## 十三、结论

成功生成：

```text
/app/share/comfyui/output/3d/ComfyUI_00001.glb
/app/share/comfyui/output/3d/ComfyUI_00002.glb
```

检查结果：

- GLB 2.0 header 正确
- 文件声明长度与实际长度一致
- bufferView 全部未越界
- accessor 有效
- mesh index 未越界
- POSITION 无 NaN / Inf
- NORMAL 无 NaN / Inf
- UV 范围正常
- tangent 正常
- 三张内嵌 PNG texture 均有效
- PBR material、metallic/roughness、normal texture 均存在

因此生成文件本身是正常的。

## 十四、3D Viewer “加载模型出错”

### 现象

旧前端在媒体资产里打开：

```text
3d/ComfyUI_00001.glb
```

提示：

```text
加载模型出错
```

### 根因

旧版本：

```text
comfyui-frontend-package 1.52.7
```

对 3D asset subfolder 解析有问题。

正确请求应拆分为：

```text
filename=ComfyUI_00001.glb
subfolder=3d
type=output
```

而错误逻辑会把：

```text
3d/ComfyUI_00001.glb
```

整体作为 filename，导致 HTTP 404。

正确请求实测：

```text
HTTP 200
Content-Type: model/gltf-binary
```

### 修复

Frontend 升级：

```text
1.52.7 -> 1.54.8
```

当前：

```text
ComfyUI core              0.37.4
comfyui-frontend-package  1.54.8
PyTorch                   2.13.0+rocm10.0.0
```

只升级了 frontend，没有升级 ROCm、Torch 或 ComfyUI core。

## 十五、备份

Trellis2 修复备份：

```text
/app/share/backup/nodes_trellis2_before_multigpu_vae_device_20260929-225617.py
/app/share/backup/nodes_trellis2_before_host_staged_crossgpu_20260929-233809.py
/app/share/backup/nodes_trellis2_before_vae_device_scope_20260929-235234.py
```

Frontend 备份：

```text
/app/share/backup/comfyui_frontend_package_1.52.7_before_1.54.8_20260930-075435.tar.gz
```

## 十六、最终生产基线

```text
ComfyUI                   0.37.4
Frontend                  1.54.8
PyTorch                   2.13.0+rocm10.0.0
Main GPU                  R9700 / cuda:0
Secondary GPU             RX 7900 XTX / cuda:1
Pixal3D                   BF16
Shape VAE                 BF16 / cuda:1
Texture VAE               BF16 / cuda:1
Structure CFG Split       2 GPUs
Shape CFG Split           2 GPUs
Target Resolution         已验证 1024 / 1536
```

## 十七、结论

Pixal3D 在 R9700 + 7900 XTX 异构双卡环境下可以稳定运行。

关键不是遇到 ROCm 错误就退回单卡，而是补全 Trellis2 / Pixal3D 的多 GPU device routing：

1. VAE 节点使用 `vae.device`
2. 异构 GPU 的关键 tensor 交接采用 host-staged transfer
3. VAE 运算显式进入正确的 `torch.cuda.device()` scope
4. 保留双卡 CFG Split
5. Frontend 升级解决 3D asset subfolder 加载错误

最终已经通过 1024 和 1536 两轮完整生成测试，并成功输出有效 GLB。

## 十八、后续维护注意事项

1. 更新 ComfyUI core 后重点检查：

```text
comfy_extras/nodes_trellis2.py
```

是否覆盖了本地补丁。

2. 若上游已经正式支持 `SelectVAEDevice`，优先使用官方逻辑；若仍存在写死 `get_torch_device()` 的地方，则需要重新核对。

3. 不要随意删除：

```text
SelectModelDevice
SelectVAEDevice
MultiGPU_WorkUnits
```

它们属于当前生产双卡结构。

4. 如果 ROCm 再崩，检查：

```text
/app/share/comfyui/app/gpucore.*.gpu
```

这些文件可能非常大。

5. Frontend 升级后浏览器若仍显示旧 UI，使用：

```text
Ctrl + Shift + R
```

强制刷新。

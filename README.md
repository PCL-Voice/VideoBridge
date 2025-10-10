# 视频语音翻译与合成工具

本项目提供了一个集**视频语音提取**、**语音翻译**和**语音合成 (Text-to-Speech, TTS)** 于一体的解决方案，旨在实现视频内容的跨语言转换。
## 环境安装
To install the required packages for running PALM-H3, please use the following command:
```bash
conda create -n <env_name> python >=3.13.7
conda activate <env_name>
pip install -r requirements.txt
```

---

## 核心文件概览

项目主目录下包含以下两个关键 Python 文件：

| 文件名 | 功能描述 | 备注 |
| :--- | :--- | :--- |
| **`F5tts.py`** | 实现了第三方 **F5-TTS 模型**的加载和推理逻辑。 | 代码引用自 [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS.git)。 |
| **`video_mt.py`** | **项目主文件**，负责级联调用所有模型，完成视频语音提取、翻译和合成的整个流程。 | 运行项目的核心脚本。 |

---

## 模型与配置

`video_mt.py` 文件通过 **`arguments`** 函数来设置所有必要的模型路径和运行参数。

### 1. 模型路径参数

这些参数**必须**指向您本地下载的模型文件路径。

| 参数名称 | 对应的模型/组件 | 功能描述 | ⚠️ 注意事项 |
| :--- | :--- | :--- | :--- |
| `--whisper_model_path` | **Whisper-large-v3** | 用于提取视频中的**语音特征** (Encoder)。 | 路径需根据您下载的模型位置进行修改。|
| `--llm_model_path` | **GemmaX2-28-9B-v0.1** | 用于将语音和文本特征转化为**目标语言的文本**。 | 路径需根据您下载的模型位置进行修改。|
| `--ckpt_path` | LoRA 微调参数 | 语音翻译任务的**微调参数**路径。 | 路径需根据您下载的模型位置进行修改。|
| `--tts_model_path` | **F5TTS\_v1\_Base** | 文本转化为**语言**。 | 路径需根据您下载的模型位置进行修改。|
| `--vocoder_local_path` | **vocos-mel-24khz** | **声码器**，用于将梅尔频谱图转化为最终的**语音**。 | 路径需根据您下载的模型位置进行修改。|

### 2. 其他运行参数

| 参数名称 | 描述 |
| :--- | :--- |
| `--cache-dir` | 在视频翻译过程中**中间文件**和最终**生成文件**的保存路径。 |

---

## 3. 主函数参数（`main`）

`video_mt.py` 的主函数接受以下关键参数：

| 参数名称 | 描述 | 示例/格式 |
| :--- | :--- | :--- |
| `video_name` | 待处理的**视频文件名**。 | `example_video.mp4` |
| `video_file` | **视频文件**的完整路径。 | `/data/videos/example_video.mp4` |
| `prompt` | 定义翻译方向的**语种标签**。 | `<zho><eng>` (将中文音频翻译为英文音频) |
| `output_path` | **冗余参数** | |

---

## 运行指南

您可以选择以下两种方式运行 `video_mt.py` 脚本：

### 方式一：使用文件内预设参数

如果所有模型路径和参数已在 `video_mt.py` 文件内部设置完毕：

```bash
python video_mt.py
```

### 方式二：命令行参数设定

可以通过命令行参数灵活地指定模型路径、缓存目录和处理的视频文件等。
``` bash
python video_mt.py \
    --cache-dir [生成文件保存的路径] \
    --whisper_model_path [whisper-large-v3模型的路径] \
    --llm_model_path [GemmaX2-28-9B-v0.1模型的路径] \
    --ckpt_path [语音翻译的lora微调参数的路径] \
    --tts_model_path [F5TTS_v1_Base 模型的路径] \
    --vocoder_local_path [vocos-mel-24khz模型的路径]
```
<img width="73" height="1531" alt="image" src="https://github.com/user-attachments/assets/b3879180-ba97-4d08-8f54-8cf1848c0857" />

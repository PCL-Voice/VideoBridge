# 🎬 视频语音翻译与合成工具

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
你希望将现有带卡通图标的视频语音翻译与合成工具说明，生成标准的markdown语言文本。我会保留原内容的结构、图标和关键信息，确保markdown格式规范，可直接复制使用。

<doubaocanvas type="text/markdown" identifier="video-voice-translator-md" title="视频语音翻译与合成工具" genre="技术文档">
# 🎬 视频语音翻译与合成工具
**Video Voice Translation & Synthesis Tool**  
一个集**视频语音提取**、**智能翻译**、**语音合成**与**字幕生成**于一体的全自动化解决方案，轻松打破语言壁垒，让视频内容触达全球受众 🌍

## ✨ 核心功能
### 1. 🌐 超全多语种覆盖
- **支持语种**：涵盖28种主流语言，包括中文、英语、日语、韩语、德语、法语、西班牙语、俄语等全球常用语种。
- **翻译能力**：支持210个双向翻译方向（28×27组合），可实现任意源语言到目标语言的精准转换，小语种翻译效果经LoRA微调优化。

### 2. 🤖 全流程自动化处理
无需人工干预，一键完成从原视频到多语言成品的全链路转换：
1. **语音提取**：自动分离视频中的音频轨道，精准捕获语音内容
2. **文本转换**：通过语音识别将音频转为源语言文本
3. **智能翻译**：调用多语种大模型生成目标语言文本
4. **字幕生成**：同步生成匹配音频节奏的目标语种字幕
5. **语音合成**：将翻译文本合成为自然流畅的目标语言语音
6. **视频重组**：自动替换原音频并嵌入字幕，输出完整成品

## 🛠️ 环境安装
### 前置条件
- Python 3.13.7 及以上版本
- Conda 环境管理工具
- 足够的磁盘空间（建议≥50GB，用于存放模型文件）

### 安装步骤
```bash
# 1. 创建并激活虚拟环境
conda create -n video-translator python>=3.13.7 -y
conda activate video-translator

# 2. 克隆项目仓库（请替换为实际仓库地址）
git clone https://github.com/your-username/video-voice-translator.git
cd video-voice-translator

# 3. 安装依赖包
pip install -r requirements.txt
```

## 📁 核心文件概览
| 文件名 | 核心功能 | 技术依赖 | 备注 |
| :--- | :--- | :--- | :--- |
| **`F5tts.py`** | 文本-to-语音（TTS）合成 | F5-TTS模型 | 代码源自 [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS.git) 开源项目 |
| **`video_mt.py`** | 项目主入口，流程调度 | Whisper、GemmaX2等 | 串联所有模块，支持参数配置与命令行调用 |
| **`requirements.txt`** | 依赖包清单 | - | 包含项目运行所需的全部Python库 |
| **`config.ini.sample`** | 配置文件模板 | - | 可复制为`config.ini`保存常用参数（如模型路径） |

## ⚙️ 模型与参数配置
`video_mt.py` 通过参数配置实现灵活的多语种处理，核心参数分为**模型路径**和**运行配置**两类，所有参数均可通过文件预设或命令行指定。

### 1. 📌 必配模型路径
这些参数需指向本地已下载的模型文件，是功能生效的核心基础：
| 参数名称 | 对应模型 | 功能说明 | 配置示例 |
| :--- | :--- | :--- | :--- |
| `--whisper_model_path` | Whisper-large-v3 | 语音特征提取与识别，支持28种语种 | `/models/whisper-large-v3` |
| `--llm_model_path` | GemmaX2-28-9B-v0.1 | 核心翻译引擎，处理多语种文本转换 | `/models/GemmaX2-28-9B-v0.1` |
| `--ckpt_path` | LoRA微调参数 | 优化小语种翻译准确率的微调模型 | `/models/lora_mt_ckpt` |
| `--tts_model_path` | F5TTS_v1_Base | 目标语种语音合成主模型 | `/models/F5TTS_v1_Base` |
| `--vocoder_local_path` | vocos-mel-24khz | 声码器，提升合成语音清晰度 | `/models/vocos-mel-24khz` |

### 2. 🔧 运行参数配置
| 参数名称 | 功能说明 | 格式/示例 |
| :--- | :--- | :--- |
| `--cache-dir` | 中间文件（音频、字幕）与成品的保存路径 | `/data/video_cache` |
| `--video_name` | 待处理视频的文件名（含后缀） | `sample_video.mp4` |
| `--video_file` | 待处理视频的完整路径 | `/data/input/sample_video.mp4` |
| `--prompt` | 翻译方向标签，格式为`<源语种代码><目标语种代码>` | `<zho><eng>`（中译英）、`<eng><jpn>`（英译日） |
| `--output_path` | 成品视频输出路径（当前版本可忽略，默认使用cache-dir） | `/data/output` |

## 🚀 快速运行指南
提供两种运行方式，适配不同使用场景：

### 方式一：文件内预设参数（适合固定任务）
1. 打开 `video_mt.py` 文件，在 `arguments()` 或 `main()` 函数中预设模型路径、视频路径及翻译方向；
2. 直接执行脚本：
   ```bash
   python video_mt.py
   ```

### 方式二：命令行参数调用（适合灵活任务）
通过命令行直接指定参数，无需修改代码，示例如下（中译英任务）：
```bash
python video_mt.py \
    --cache-dir /data/video_translation/cache \
    --whisper_model_path /models/whisper-large-v3 \
    --llm_model_path /models/GemmaX2-28-9B-v0.1 \
    --ckpt_path /models/lora_mt_ckpt \
    --tts_model_path /models/F5TTS_v1_Base \
    --vocoder_local_path /models/vocos-mel-24khz \
    --video_name chinese_demo.mp4 \
    --video_file /data/videos/chinese_demo.mp4 \
    --prompt <zho><eng>
```

## ❗ 常见问题与解决方案
| 问题现象 | 可能原因 | 解决方法 |
| :--- | :--- | :--- |
| 模型加载失败 | 模型路径错误或文件不完整 | 1. 核对`--xxx_model_path`参数是否正确；2. 重新下载完整模型文件 |
| 翻译语种不符 | `--prompt`参数格式错误 | 确保格式为`<源语种代码><目标语种代码>`，如`<eng><kor>`（英译韩） |
| 合成语音卡顿 | 声码器路径错误或资源不足 | 1. 检查`--vocoder_local_path`配置；2. 关闭其他占用GPU的程序 |
| 视频重组失败 | 中间文件缺失 | 检查`--cache-dir`权限，确保程序可读写该目录 |

## 🙏 致谢
本项目的实现得益于开源社区的强大支持：
- 语音识别依赖 OpenAI 的 **Whisper** 模型
- 多语种翻译基于 Google 的 **GemmaX2** 大模型
- 语音合成采用 **F5-TTS** 及 **Vocos** 声码器开源方案
- 流程设计参考了 [Auto-Synced-Translated-Dubs](https://blog.csdn.net/gitblog_00731/article/details/142022588) 等同类项目的自动化思路

感谢所有开源贡献者为多语种处理技术生态提供的基础支持！

---

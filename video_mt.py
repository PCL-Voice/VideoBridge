import argparse
import os
import torch
import json
from torch import nn
from pathlib import Path
from moviepy.audio.io.AudioFileClip import AudioFileClip
from fsmnvad import FSMNVad
import librosa
import soundfile as sf
from F5tts import F5TTS
from peft import LoraConfig, PeftType, get_peft_model
from transformers import (
    PreTrainedModel,
    WhisperModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
    WhisperConfig,
)

class EncoderProjectorQFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_dim = 1280
        self.llm_dim = 3584
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = 8

        self.query_len = 80
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        if self.llm_dim <= 1536:
            self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)
        elif self.llm_dim <= 2560:
            self.linear1 = nn.Linear(configuration.hidden_size, 1536)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(1536, self.llm_dim)
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)
        else:
            self.linear1 = nn.Linear(configuration.hidden_size, 2560)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(2560, self.llm_dim)
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)

        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )

        if self.llm_dim <= 1536:
            query_proj = self.norm(self.linear(query_output.last_hidden_state))
        else:
            x = self.linear1(query_output.last_hidden_state)
            x = self.relu(x)
            x = self.linear2(x)
            query_proj = self.norm(x)

        return query_proj


class CustomSLM(PreTrainedModel):
    def __init__(self, config, whisper_model_path, llm_model_path, ckpt_path=None):
        super().__init__(config)

        self.audio_processor = WhisperFeatureExtractor.from_pretrained(whisper_model_path)
        self.encoder = WhisperModel.from_pretrained(whisper_model_path, torch_dtype=torch.float16).encoder
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path, torch_dtype=torch.float16)
        peft_config = LoraConfig(
            peft_type=PeftType.LORA,  # 直接使用枚举值 PeftType.LORA，不需要 <...> 符号
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            target_modules=["q_proj", "v_proj"],  # 使用列表而非集合（set）
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            # 其他参数若不需要自定义，可以省略（库会使用默认值）
        )
        self.llm = get_peft_model(self.llm, peft_config).to(torch.float16)
        self.encoder_projector = EncoderProjectorQFormer().to(torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        #
        if ckpt_path is not None:
            print("loading model checkpoint from: {}".format(ckpt_path))
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(ckpt_dict, strict=False)  #

    @torch.no_grad()
    def inference(
            self,
            audios=None,
            prompt=None,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=None,
            assistant_model=None,
            streamer=None,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
            **kwargs,
    ):
        # inference for asr model

        if audios is not None:  # Audio-Text QA

            audio_mel = self.audio_processor(audios, return_tensors="pt").input_features
            # audio_mel = audio_mel.T  # (time_steps, n_mels)
            audio_mel = audio_mel.to(dtype=torch.float16).to(self.encoder.device)
            encoder_outs = self.encoder(audio_mel).last_hidden_state

            audio_mel_post_mask = torch.ones(
                encoder_outs.size()[:-1], dtype=torch.long
            ).to(encoder_outs.device)
            encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
        else:  # Text QA
            encoder_outs = torch.empty(
                1, 0, self.llm.model.embed_tokens.embedding_dim
            )#.to(device)

        prompt = prompt
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(self.llm.device)
        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)
        inputs_embeds = inputs_embeds.unsqueeze(0).repeat(10,1, 1)
        inputs_embeds = torch.cat(
            (encoder_outs, inputs_embeds), dim=1
        )  # [audio,prompt]
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
            inputs_embeds.device
        )
        # generate
        with torch.autocast(device_type='cpu', dtype=torch.float16):
            model_outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=kwargs.get("max_new_tokens", 300),
                num_beams=kwargs.get("num_beams", 1),
                do_sample=kwargs.get("do_sample", False),
                min_length=kwargs.get("min_new_tokens", 10),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=kwargs.get("repetition_penalty", 1),
                length_penalty=kwargs.get("length_penalty", 1.0),
                temperature=kwargs.get("temperature", 1.0),
                no_repeat_ngram_size=5,
                early_stopping=True,
                attention_mask=attention_mask,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        llm_outputs = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
        return llm_outputs

def writer_srt(srt_file, segments):
    with open(srt_file, "w", encoding="utf-8") as srt_writer:
        for i, (start_time, end_time) in enumerate(segments):
            start_time = start_time / 1000
            end_time = end_time / 1000
            start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"
            srt_writer.write(f"{i}\n")
            srt_writer.write(f"{start_time_str} --> {end_time_str}\n")
            srt_writer.write(f"语音片段 {i}\n\n")

class VideoMT():
    def __init__(
            self,
            args,
            cache_dir="cache"
    ):
        super().__init__()
        self.vad = FSMNVad()
        self.batch_size = 10
        self.cache_dir = Path(cache_dir)
        config = WhisperConfig.from_pretrained(args.whisper_model_path)
        self.audiomt = CustomSLM(config, args.whisper_model_path, args.llm_model_path, ckpt_path=args.ckpt_path)
        self.tts = F5TTS(
            model=args.tts_model_path,
            vocoder_local_path=args.vocoder_local_path,
            ckpt_file=os.path.join(args.tts_model_path, "model_1250000.safetensors"),
        )

    def extract_and_cache_audio(self, video_path, sr=16000):
        """
        提取视频文件的音频并缓存到本地，转换为16kHz的WAV格式

        参数:
            video_path (str): 视频文件路径
            cache_dir (str): 缓存目录，默认为"cache"

        返回:
            str: 缓存的音频文件路径，如果失败则返回None
        """
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 获取视频文件名（不含扩展名）
        video_filename = os.path.splitext(os.path.basename(video_path))[0]

        # 构建缓存文件路径（同名但扩展名为.wav）
        cached_audio_path = os.path.join(self.cache_dir, f"{video_filename}.wav")

        # 如果已经存在缓存，直接返回
        if os.path.exists(cached_audio_path):
            print(f"使用缓存音频: {cached_audio_path}")
            audio, rate = librosa.load(cached_audio_path, sr=sr)
            return cached_audio_path, audio

        # 使用ffmpeg提取音频并转换为16kHz WAV
        try:
            audio = AudioFileClip(video_path)
            audio.write_audiofile(cached_audio_path)
            audio, rate = librosa.load(cached_audio_path, sr=sr)
            sf.write(cached_audio_path, audio, sr)
            print(f"音频已提取并缓存到: {cached_audio_path}")
            return cached_audio_path, audio
        except Exception as e:
            print(f"提取音频失败: {e}")
            return None, None, None

    def split_audio(self, video_name, segments, prompt, audio, output_path, sr=16000):
        audio_list, audio_names, durations = [], [], []
        for i, (start_time, end_time) in enumerate(segments):
            start_index = int(start_time * sr / 1000)
            end_index = int(end_time * sr / 1000)
            duration = (end_time - start_time) / 1000
            audio_i = audio[start_index:end_index]
            audio_path = self.cache_dir.joinpath(f"{video_name}_{i}.wav")
            sf.write(audio_path, audio_i, sr)
            audio_list.append(audio_i)
            audio_names.append(audio_path)
            durations.append(duration)

        llm_outputs = self.audiomt.inference(
            audios=audio_list,
            prompt=prompt,
        )
        jsonl_file = os.path.join(self.cache_dir, f"{video_name}.jsonl")
        with open(jsonl_file, "w", encoding="utf-8") as jsonl_writer:
            for i in range(len(llm_outputs)):
                if prompt in llm_outputs[i]:
                    start_time, end_time = segments[i]
                    start_time = start_time / 1000
                    end_time = end_time / 1000
                    source_text, translation_text = llm_outputs[i].split("<eos>")[0].split(prompt)
                    tts_audio_path = self.cache_dir.joinpath(f"{video_name}_tts_{i}.wav")
                    wav, sr, spec = self.tts.infer(
                        ref_file=audio_names[i],
                        ref_text=source_text,
                        gen_text=translation_text,
                        file_wave=tts_audio_path,
                    )
                    start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
                    end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"
                    jsonl_data = {
                            "audio": f"{video_name}_{i}.wav",
                            "prompt": prompt,
                            "gt": f"语音片段 {i}<|zho|><|eng|>语音片段 {i}",
                            "source": "video_zho_eng",
                            "index": i,
                            "startTime": start_time_str,
                            "endTime": end_time_str,
                            "startTimeInSeconds": f"{int((start_time % 1) * 1000):03}",
                            "endTimeInSeconds": f"{int((end_time % 1) * 1000):03}",
                            "duration": durations[i],
                            "response": "111", ##llm_outputs,
                            "audio_path": str(tts_audio_path),
                            "sourceText": source_text,
                            "targetText": translation_text
                        }

                    jsonl_writer.write(f"{json.dumps(jsonl_data, ensure_ascii=False)}\n")


    def processing_video(self, video_name, prompt, video_path, output_path):
        audio_path, audio = self.extract_and_cache_audio(video_path)
        segments = self.vad.segments_offline(audio_path)
        srt_file = os.path.join(self.cache_dir, f"{video_name}.srt")
        if not os.path.exists(srt_file):
            writer_srt(srt_file, segments)
        self.split_audio(video_name, segments, prompt, audio, output_path)

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="./examples")
    parser.add_argument("--whisper_model_path", type=str, default=r"E:\pred_model\whisper-large-v3")
    parser.add_argument("--llm_model_path", type=str, default=r"E:\data\videomt\models\GemmaX2-28-9B-v0.1")
    parser.add_argument("--ckpt_path", type=str, default=r"E:\data\videomt\models\asr_epoch_1_step_9000/model.pt")
    parser.add_argument("--tts_model_path", type=str, default=r"E:\data\videomt\models\F5TTS_v1_Base")
    parser.add_argument("--vocoder_local_path", type=str, default=r"E:\data\videomt\models\vocos-mel-24khz")

    args = parser.parse_args()
    return args
def main():
    # 示例用法
    video_name = "xinwen_60"
    prompt = "<|zho|><|eng|>"
    video_file = f"./{video_name}.webm"  # 替换为你的视频文件路径
    output_path = "./test_dir"
    args = arguments()
    videomt = VideoMT(args, cache_dir=args.cache_dir)
    videomt.processing_video(video_name, prompt, video_file, output_path)


if __name__ == '__main__':

    main()
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Example of how to synthesize with the XTTS model
# text: The text to synthesize
# speaker_wav: The path to the speaker audio file
# output_path: The path to save the output audio file

class xtts_model:
    def __init__(self):
        print("Loading model...")
        self.config = XttsConfig()
        self.config.load_json(r"audioGen\checkpoint\config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_path=r"audioGen\checkpoint\best_model.pth", checkpoint_dir=r"audioGen\checkpoint")
        self.model.cuda()

    def synthesis(self, text, speaker_wav, output_path):
        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[speaker_wav])

        print("Inference...")
        out = self.model.inference(
            text,
            "zh-cn",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7, # Add custom parameters here
        )
        torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)


if __name__ == "__main__":
    model = xtts_model()
    model.synthesis("生成測試。", r"audioGen\reference\speaker\speaker_bryan.wav", "1.wav")
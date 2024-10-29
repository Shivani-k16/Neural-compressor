import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration,pipeline
from datasets import load_dataset
import time
import torchaudio
import intel_extension_for_pytorch

print(f"PyTorch Version: {torch.__version__}")

device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
#device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Inference before quantization

model_id='openai/whisper-small'
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,use_safetensors=True
)
model.to(device)

processor = WhisperProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

start_time = time.time()

result = pipe(sample)

end_time=time.time()
print(result["text"])
print(f"Time spent :{end_time-start_time} seconds")


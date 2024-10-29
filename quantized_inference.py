from neural_compressor.utils.pytorch import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import os
import torch
import time


model_id='openai/whisper-small'
model = WhisperForConditionalGeneration.from_pretrained(model_id,use_safetensors=True)

# Load the quantized model
model = load(os.path.abspath(os.path.expanduser('./quantized_whisper')), model)

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# generate token ids
start_time = time.time()

predicted_ids = model.generate(input_features)

end_time=time.time()
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
print(f"Time spent after quantization :{end_time-start_time} seconds")


from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.utils.pytorch import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration


model_id='openai/whisper-small'
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,use_safetensors=True)
output_dir = "quantized_whisper"

tune=True

if tune:
    tuning_criterion = TuningCriterion(max_trials=5)
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.1)
    op_type_dict = {
        "Embedding": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}
        }
    conf = PostTrainingQuantConfig(approach="dynamic", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion,op_type_dict=op_type_dict)
    q_model = quantization.fit(model, conf=conf) 
    q_model.save(output_dir) 
    

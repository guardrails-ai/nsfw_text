from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import pipeline
import os
import torch

app = FastAPI()

device = os.environ.get("GUARDRAILS_DEVICE", "cpu")

if device == "cuda" and torch.cuda.is_available():
    torch_device = "cuda"
elif device == "cuda" and not torch.cuda.is_available():
    print("Warning: CUDA is not available. Falling back to CPU.")
    torch_device = "cpu"
else:
    torch_device = "cpu"
    
class InferenceData(BaseModel):
    name: str
    shape: List[int]
    data: List
    datatype: str

class InputRequest(BaseModel):
    inputs: List[InferenceData]

class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[InferenceData]

@app.get("/")
async def hello_world():
    return "nsfw_text"

@app.post("/validate", response_model=OutputResponse)
async def check_nsfw(input_request: InputRequest):
    threshold = None
    for inp in input_request.inputs:
        if inp.name == "text":
            text_vals = inp.data
        elif inp.name == "threshold":
            threshold = float(inp.data[0])

    if text_vals is None or threshold is None:
        raise HTTPException(status_code=400, detail="Invalid input format")

    return NSFWText.infer(text_vals, threshold)

class NSFWText:
    model_name = "michellejieli/NSFW_text_classifier"
    pipe = pipeline(
        "text-classification", 
        model=model_name,
        device=torch_device
    )

    def infer(text_vals, threshold) -> OutputResponse:
        outputs = []
        for idx, text in enumerate(text_vals):
            results = NSFWText.pipe(text)
            pred_labels = [
                result['label'] for result in results if result['label'] == 'NSFW' and result['score'] > threshold
            ]
            outputs.append(
                InferenceData(
                    name=f"result{idx}",
                    datatype="BYTES",
                    shape=[len(pred_labels)],
                    data=[pred_labels],
                )
            )

        output_data = OutputResponse(
            modelname=NSFWText.model_name, modelversion="1", outputs=outputs
        )

        return output_data

# Run the app with uvicorn
# Save this script as app.py and run with: uvicorn app:app --reload
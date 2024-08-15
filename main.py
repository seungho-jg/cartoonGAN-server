import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from onnx_cartoongan_model import load_model, cartoon_gan_inference

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", "cartoongan_model.onnx")

# 모델 로드
try:
    model = load_model(model_path)
    print(f"ONNX 모델이 성공적으로 로드되었습니다. 모델 경로: {model_path}")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {str(e)}")
    model = None

@app.post("/api/convert")
async def convert_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)

        # 모델 추론
        output_image = cartoon_gan_inference(model, image_np)

        # NumPy 배열을 PIL Image로 변환
        result_image = Image.fromarray(output_image)

        # 결과 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # 스트리밍 응답으로 이미지 반환
        return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
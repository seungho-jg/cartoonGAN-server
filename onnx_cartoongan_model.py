import onnxruntime as ort
import numpy as np

def load_model(model_path):
    return ort.InferenceSession(model_path)

def preprocess_image(image):
    # 이미지를 float32로 변환하고 [0, 1] 범위로 정규화
    image = image.astype(np.float32) / 255.0
    # 모델 입력 형태에 맞게 차원 추가 (배치 크기 1)
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_image(image):
    # [0, 1] 범위의 이미지를 [0, 255] 범위로 변환
    image = image * 255
    # uint8로 변환하고 배치 차원 제거
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.squeeze(image, axis=0)
    return image

def cartoon_gan_inference(session, input_image):
    preprocessed_image = preprocess_image(input_image)
    
    # 모델의 입력 이름 가져오기
    input_name = session.get_inputs()[0].name
    
    # 추론 실행
    output = session.run(None, {input_name: preprocessed_image})
    
    # 첫 번째 출력 결과 사용
    output_image = output[0]
    
    return postprocess_image(output_image)
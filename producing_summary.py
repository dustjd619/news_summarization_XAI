import json
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# .env 파일 로드
load_dotenv()

# Hugging Face 토큰 환경 변수 로드
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=huggingface_token)

# 모델 및 토크나이저 로드, 아래 model_name수정하면 다른 모델로 요약문 생성 가능
model_name = "google/gemma-3-27b-it" 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

# 모델 로딩: GPU로 전체 모델을 로드 (device_map="auto" 대신 직접 명시적으로 GPU로 설정) /colab에서 필요했음
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

def load_json(file_path):
    """JSON 파일을 로드하는 함수"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def generate_summary(original_text, annotations, use_annotations=True):
    """LLM을 이용하여 요약문을 생성하는 함수"""
    
    annotations_text = ""
    for keyword, info in annotations.items():
        if isinstance(info, dict):
            annotation = info.get("annotation", "")
            importance = info.get("importance", 0.5)
            annotations_text += f"{keyword}: {annotation} (중요도: {importance})\n"


    if use_annotations:
        # LLM 입력 프롬프트 생성 (주석과 중요도 사용) 추가로 점수별 반영 가이드라인을 제시할지 고민
        prompt = (
            f"기사 내용을 기반으로 핵심 내용을 요약하세요. 지침:아래 제공된 키워드와 각 키워드의 중요도 점수를 고려하여 요약하세요. \n\n"
            f"기사 내용: {original_text}\n\n"
            f"관련 개념 설명 (중요도 포함):\n{annotations_text}\n\n"
            f"요약:"
        )
    else:
        # LLM 입력 프롬프트 생성 (주석과 중요도 없이)
        prompt = (
            f"기사 내용을 기반으로 핵심 내용을 요약하세요.\n\n"
            f"기사 내용: {original_text}\n\n"
            f"요약:"
        )
    
    # LLM 입력 토큰화
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 모델과 입력을 동일한 디바이스로 이동 /colab에서 필요했음
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # 모델을 이용해 요약 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,  # Explicitly limit new tokens
            do_sample=True,
            top_p=0.95,
            temperature=0.1,
            num_return_sequences=1
        )
    
    # 생성된 요약 디코딩
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 프롬프트 부분 제외하고 실제 요약문만 반환
    summary_start = summary.find("요약:")
    if summary_start != -1:
        summary = summary[summary_start + len("요약:"):].strip()  # "요약:" 이후의 부분만 추출
    
    return summary.strip()

def process_documents(data):
    """각 문서에 대해 요약을 생성하고 JSON에 저장하는 함수"""
    for document in data['documents']:
        print(f"Processing document {document['id']}...")

        original_text = document.get('original_text', '')
        annotations = document.get('annotations', {})
        
        # annotations를 이용한 요약문 생성
        augmented_summary = generate_summary(original_text, annotations, use_annotations=True)
        
        # annotations를 사용하지 않은 요약문 생성
        plain_summary = generate_summary(original_text, annotations, use_annotations=False)
        
        # 요약 결과를 기존 필드에 저장
        document['summary_with_annotations'] = augmented_summary
        document['summary_without_annotations'] = plain_summary
    
    return data

#하단 - 실행 과정

# 파일 경로 설정
input_file = "modified_dataset.json"  # 수정된 데이터셋 파일
output_file = "final_dataset.json"  # 최종 결과 파일

# 데이터 로드
data = load_json(input_file)

# 문서 처리 및 요약 생성
data = process_documents(data)

# 결과 저장
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Updated JSON saved to: {output_file}")
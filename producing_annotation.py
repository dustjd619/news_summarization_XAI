import json
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# .env 파일 로드
load_dotenv()

# Hugging Face 토큰 환경 변수로 불러오기
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Hugging Face에 로그인 (토큰 사용)
login(token=huggingface_token)

# Gemma 모델과 토크나이저 로드
model_name = "google/gemma-3-27b-it"  # 사용 모델 - 수정하면 됨
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# 현재 모델이 위치한 장치
device = model.device  # 모델이 있는 장치 (GPU 또는 CPU)

# annotation_list.json을 로드하거나 파일이 없으면 초기화
def load_annotation_list(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            annotations_data = json.load(file)
    else:
        annotations_data = {}
    return annotations_data

# JSON 파일 로드 함수
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 키워드 추출 함수
def extract_keywords(document):
    # 새로운 JSON에서 annotations 부분을 키워드로 사용
    return [{"word": word, "importance": document['annotations'][word]['importance']} for word in document.get('annotations', {}).keys()]

# 주석 생성 함수 (LLM 호출 포함)
def generate_annotations(keyword):
    # 경제 뉴스 맥락에 맞춘 프롬프트 생성
    prompt = f"'{keyword}'은/는 경제적 관점에서 중요한 개념으로, 그 의미와 경제적 의의를 간략하게 한두 문장 이내로 설명해주세요."
    
    # 입력 텍스트 토큰화
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 모델을 사용해 주석 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.95,
            temperature=0.1,
            max_length=1024,
            num_return_sequences=1
        )

    # 생성된 주석을 디코딩
    annotation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    annotation = annotation.replace(prompt, "").strip()

    # 주석 반환
    return annotation

# 파일 경로 설정
file_path = "processed_econ_news_train.json"
output_path = "modified_dataset.json"  # 주석이 추가된 파일을 저장할 경로
annotation_file_path = "annotation_list.json"

# annotation_list.json을 한 번만 로드
annotations_data = load_annotation_list(annotation_file_path)

# JSON 파일 로드
data = load_json(file_path)

# 각 document_id별로 작업 수행
for document in data['documents']:
    document_id = document.get('id')
    print(f"문서 ID {document_id} 처리 중...")

    # 키워드 추출
    keywords = extract_keywords(document)
    
    # 주석 생성 (LLM 호출 포함)
    annotations = {}

    for keyword in keywords:
        word = keyword.get('word', '')
        importance = keyword.get('importance', 0.5)

        # 주석이 이미 annotation_list.json에 존재하는지 확인
        if word in annotations_data:
            # 기존 주석을 가져옴
            annotations[word] = {
                "annotation": annotations_data[word],
                "importance": importance
            }
        else:
            # 없으면 LLM을 호출하여 주석 생성
            generated_annotation = generate_annotations(word)
            annotations[word] = {
                "annotation": generated_annotation,
                "importance": importance
            }

            # annotation_list.json에 새 키워드와 주석 추가 (importance 제외)
            annotations_data[word] = generated_annotation

    # document에 주석 추가
    document['annotations'] = annotations
    
    # 주석 모델 정보 업데이트
    document['annotation_model'] = model_name

    # 추가적인 정보 업데이트
    document['summary_with_annotations'] = document.get('summary_with_annotations', '')
    document['summary_without_annotations'] = document.get('summary_without_annotations', '')
    document['ground_truth_summary'] = document.get('ground_truth_summary', '')
    document['bert_score_original'] = document.get('bert_score_original', 0)
    document['bert_score_summary_without_annotations'] = document.get('bert_score_summary_without_annotations', 0)
    document['bert_score_summary_with_annotations'] = document.get('bert_score_summary_with_annotations', 0)

# annotation_list.json에 새 주석 저장
with open(annotation_file_path, 'w', encoding='utf-8') as file:
    json.dump(annotations_data, file, ensure_ascii=False, indent=4)

# 결과를 새로운 JSON 파일에 저장
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)

# 체크용
print(f"Updated JSON saved to: {output_path}")

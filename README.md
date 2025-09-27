# 📊 시맨틱 커널 및 Azure AI Foundry 기반 AI Agent 서비스

## 🖥️ 서버 환경 세팅 예시 (Ubuntu 22.04 기준 예시)

```bash
### Package 업데이트 및 Python 설치 ###
sudo apt update && sudo apt install python3 && sudo apt install python3-pip
python3 --version

### Git 설치 ###
sudo apt install git
git --version

### Git clone 실행 ###
git clone https://github.com/jungkwanpark/project2.git

### 디렉토리 이동 ###
cd project2
ls -la

### 의존성 설치 ###
pip3 install -r requirements.txt

### 환경변수 파일 (.env) 생성 (아래 예시 참조) ###
Example : 
--------------------------------------------------------------
AZURE_OPENAI_MODEL="gpt-4.1"
AZURE_OPENAI_TEMPERATURE=0.5
AZURE_OPENAI_VERSION="2025-01-01-preview"
AZURE_OPENAI_ENDPOINT="https://your-azure-openai-endpoint"
AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
AZURE_SEARCH_SERVICE_ENDPOINT="https://your-azure-ai-search-endpoint"
AZURE_SEARCH_INDEX_NAME="your-azure-ai-search-index"
AZURE_SEARCH_API_KEY="your-azure-ai-search-api-key"
AZURE_SEARCH_KEY_FIELD_NAME="your-azure-ai-search-key-field-name"
AZURE_SEARCH_CONTENT_FIELD_NAME="your-azure-ai-search-content-field-name"
--------------------------------------------------------------

### Streamlit 으로 app_semantic_kernel.py 파일 실행 (포트 8080 으로 실행) ###
python3 -m streamlit run app_semantic_kernel.py --server.port 8080

```

## 🖥️ 원격 클라이언트에서 서버 접속

```bash
http://your-server-domain:8080
```

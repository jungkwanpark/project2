import os
from dotenv import load_dotenv

import streamlit
import requests
import tempfile
import uuid 
import asyncio 
import pypdf

# 마이크로소프트 AI Agent 프레임워크인 Semantic Kernel 활용
import semantic_kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.text import text_chunker 
from semantic_kernel.functions.kernel_arguments import KernelArguments 


# .env 파일에서 환경 변수 로드
load_dotenv()

# Azure OpenAI 환경 변수 읽기
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_TEMPERATURE = float(os.getenv("AZURE_OPENAI_TEMPERATURE", 0.5))

# AI Search 환경 변수 읽기
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_KEY_FIELD_NAME = os.getenv("AZURE_SEARCH_KEY_FIELD_NAME", "id")
AZURE_SEARCH_CONTENT_FIELD_NAME = os.getenv("AZURE_SEARCH_CONTENT_FIELD_NAME", "content")


# [1] Semantic Kernel 모델 선언
kernel = semantic_kernel.Kernel()

# [1-1] Semantic Kernel 에 Azure Chat Completion 서비스 연동
kernel.add_service(
    AzureChatCompletion(
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_VERSION,
        deployment_name=AZURE_OPENAI_MODEL,
    )
)


# [2] RAG 구현

# [2-1]  Azure AI Search 인덱스 및 문서 초기화
def clear_azure_search_index():
    search_url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs?api-version=2024-07-01&search=*&$select={AZURE_SEARCH_KEY_FIELD_NAME}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    
    try: # 오류가 발생할 수 있는 코드 블록 지정
        response = requests.get(search_url, headers=headers)
        response.raise_for_status() 
        results = response.json()       
        doc_keys = [doc[AZURE_SEARCH_KEY_FIELD_NAME] for doc in results.get("value", [])]
 
        if not doc_keys:  
            print("[LOG] 인덱스가 이미 비어있습니다. 초기화를 건너뜁니다.")
            return True  

        docs_to_delete = [{"@search.action": "delete", AZURE_SEARCH_KEY_FIELD_NAME: key} for key in doc_keys]
        index_url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/index?api-version=2024-07-01"
        payload = {"value": docs_to_delete}
        
        response = requests.post(index_url, headers=headers, json=payload)
        response.raise_for_status()
        
        print(f"[LOG] {len(doc_keys)}개의 문서를 성공적으로 삭제하여 인덱스를 초기화했습니다.")
        return True

    except requests.RequestException as e: 
        print(f"[ERROR] Azure AI Search 인덱스 초기화 실패: {e}")
        return False  


# [2-2] Text 를 Load -> Split 처리
def load_and_split_pdf(file):
    """업로드된 PDF 파일에서 텍스트를 추출하고 Semantic Kernel의 TextChunker를 사용하여 분할합니다."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

        tmp_file.write(file.getvalue())    
        tmp_file_path = tmp_file.name

    try:
        reader = pypdf.PdfReader(tmp_file_path)              
        full_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        chunks = text_chunker.split_plaintext_lines(full_text, 1000)

    finally: 
        os.remove(tmp_file_path) 

    return chunks 


# [2-3] Split 된 텍스트 조각들을 Azure AI Search 에 <업로드> 하여 <인덱싱> 처리하는 함수 정의
def index_documents_to_azure_search(docs: list[str]):
    """분할된 텍스트 청크 목록을 Azure AI Search에 인덱싱합니다."""

    if docs: 
        print("\n--- [디버깅 LOG] 인덱싱될 첫 번째 문서 청크 ---")
        print(docs[0])
        print("------------------------------------------------\n")

    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/index?api-version=2024-07-01"

    documents_to_upload = [
        {"@search.action": "upload", AZURE_SEARCH_KEY_FIELD_NAME: str(uuid.uuid4()), AZURE_SEARCH_CONTENT_FIELD_NAME: doc} for doc in docs
        ]
    
    payload = {"value": documents_to_upload}

    try: 
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() 
        print(f"[LOG] {len(docs)}개의 문서 청크를 성공적으로 인덱싱했습니다.")
        return True
    
    except requests.RequestException as e: 
        error_details = ""
        try: 
            if e.response is not None:
                error_details = e.response.json() 
        except Exception: # 
            error_details = "Could not parse error response from Azure AI Search."        
        print(f"[ERROR] Azure AI Search 인덱싱 실패: {e}")
        print(f"[ERROR] 상세 정보: {error_details}")

        return False


# [2-4] 사용자의 질문(query)을 바탕으로 Azure AI Search 에서 문서 검색 (Azure AI Search 로 RAG 질의)
def search_azure_ai(query: str) -> str: 
    """사용자 질문과 관련된 문서를 Azure AI Search에서 검색합니다."""

    url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}    
    payload = {
        "search": query,
        "count": True,
        "top": 5,
        "searchFields": AZURE_SEARCH_CONTENT_FIELD_NAME
    }

    try: 
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() 
        results = response.json() 
        
        print("\n--- [디버깅 LOG] Azure AI Search 검색 결과 ---")
        print(results)
        print("---------------------------------------------\n")

    except requests.RequestException as e: 
        print(f"[ERROR] Azure AI Search 요청 실패: {e}")
        return "검색 서비스 호출에 실패했습니다. CLI 콘솔의 에러 로그를 확인해주세요."

    return "\n\n".join([doc.get(AZURE_SEARCH_CONTENT_FIELD_NAME, "") for doc in results.get("value", [])])


# [2-5] RAG 의 System Prompt 템플릿 정의
qa_system_prompt = """
당신은 질문에 답변하는 AI 어시턴트입니다.
제공된 컨텍스트 정보를 사용하여 질문에 답변해 주세요.
만약 컨텍스트에서 답을 찾을 수 없다면, "정보를 찾을 수 없습니다."라고 솔직하게 말해주세요.
답변은 항상 한국어로, 친절한 존댓말로 작성해야 합니다.

---
[컨텍스트]:
{{$context}}
---

[질문]:
{{$input}}

[답변]:
"""


# [3] Streamlit (채팅창) 로직 구현

# [3-1] Streamlit 타이틀 정의
streamlit.title("Azure OpenAI 챗봇 서비스 (Semantic Kernel) 💬")

# [3-2] Streamlit 사이드바 (PDF 파일 업로드) 구현 
with streamlit.sidebar: 

    streamlit.header("RAG 기능 활성화")
    uploaded_file = streamlit.file_uploader("PDF 파일을 업로드하면 해당 내용으로 RAG 챗봇이 동작합니다.", type=["pdf"])

    if uploaded_file is not None: # Upload 된 파일이 존재할 경우
        if "last_uploaded_file" not in streamlit.session_state or streamlit.session_state.last_uploaded_file != uploaded_file.name:
            with streamlit.status("파일 처리 중...", expanded=True) as status: 
                status.write("기존 인덱스 데이터를 초기화합니다...") 
                if clear_azure_search_index(): 
                    status.write("PDF 파일을 분석하고 있습니다...") 
                    split_docs = load_and_split_pdf(uploaded_file)
                    status.write(f"파일을 {len(split_docs)}개의 청크로 분할했습니다.") 
                    status.write("분석된 내용을 Azure AI Search에 저장하고 있습니다...") 
                    if index_documents_to_azure_search(split_docs):
                        status.update(label="파일 처리 완료!", state="complete", expanded=False)                   
                        streamlit.session_state.rag_enabled = True
                        streamlit.session_state.last_uploaded_file = uploaded_file.name
                        if "messages" in streamlit.session_state: 
                            streamlit.session_state.messages.append(
                                {"role": "assistant", "content": f"✅ **{uploaded_file.name}** 파일의 내용을 성공적으로 학습했습니다."}
                            )
                        streamlit.rerun()

                    else:
                        status.update(label="파일 처리 실패", state="error", expanded=True)
                        streamlit.error("파일 내용을 AI Search에 저장하는 데 실패했습니다. CLI 로그를 확인해주세요.")

                else: 
                    status.update(label="인덱스 초기화 실패", state="error", expanded=True)
                    streamlit.error("기존 인덱스 데이터를 삭제하는 데 실패했습니다. CLI 로그를 확인해주세요.")


# [3-3] Streamlit 최초 화면 인사말
if "messages" not in streamlit.session_state:  
    streamlit.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다."}]


# [3-4] 과거 대화 내역 표시 (채팅 History 표시)
for msg in streamlit.session_state.messages:  
    streamlit.chat_message(msg["role"]).write(msg["content"])


# [3-5] 사용자 입력 처리
if prompt := streamlit.chat_input("질문을 입력해주세요..."):

    streamlit.session_state.messages.append({"role": "user", "content": prompt})
    streamlit.chat_message("user").write(prompt)

    with streamlit.chat_message("assistant"): 
        with streamlit.spinner("답변을 생성하는 중입니다..."): 
            try: 
                if streamlit.session_state.get("rag_enabled", False):
                    context = search_azure_ai(prompt) 
                    arguments = KernelArguments(input=prompt, context=context) 
                    result = asyncio.run(
                        kernel.invoke_prompt(
                            prompt=qa_system_prompt, 
                            arguments=arguments)
                    )
                    response = str(result)              
                else: 
                    result = asyncio.run(kernel.invoke_prompt(prompt))
                    response = str(result)
                streamlit.session_state.messages.append({"role": "assistant", "content": response})
                streamlit.write(response)

            except Exception as e: 
                streamlit.error(f"답변 생성 중 오류가 발생했습니다: {e}") 
                print(f"[ERROR] Kernel invocation failed with exception: {e}")
                import traceback 
                traceback.print_exc() 



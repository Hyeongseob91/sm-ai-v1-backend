# AI-Backend 리팩토링 문서

## 개요

기존 `ai_backend/` 구조를 새로운 모듈식 아키텍처 `ai-backend/`로 리팩토링하였습니다.
이 문서는 새로운 아키텍처의 구조, 각 모듈의 역할, 그리고 사용 방법을 설명합니다.

---

## 아키텍처 목표

1. **명확한 계층 분리**: API, Core, Systems로 역할 분리
2. **모듈성**: 각 시스템(RAG, Chat)이 독립적으로 동작
3. **확장성**: 새로운 시스템 추가 용이
4. **재사용성**: 공통 컴포넌트 공유

---

## 폴더 구조

```
ai-backend/
├── main.py                              # FastAPI 앱 엔트리포인트
├── pyproject.toml                       # Poetry 의존성 정의
│
└── src/
    ├── api/                             # 🌐 API 계층
    │   ├── __init__.py
    │   ├── router.py                    # 라우터 등록 및 관리
    │   ├── chat_endpoints.py            # 채팅 API 엔드포인트
    │   └── rag_endpoints.py             # RAG API 엔드포인트
    │
    ├── config/                          # 🔑 설정 계층
    │   ├── __init__.py
    │   ├── config_model.py              # LLM 모델, API 키, 경로 설정
    │   ├── config_prompts.py            # 프롬프트 디렉토리 경로
    │   └── config_db.py                 # 벡터 DB 연결 설정
    │
    ├── core/                            # 🏭 Core 계층
    │   ├── __init__.py
    │   ├── graph_factory.py             # ⭐ 시스템 선택에 따른 체인 생성
    │   ├── session_manager.py           # 세션/대화 기록 관리
    │   └── llm_service.py               # LLM 클라이언트 통합 관리
    │
    ├── data/                            # 💾 데이터 계층
    │   └── prompts/
    │       ├── chatbot/                 # 채팅봇 프롬프트
    │       ├── rag/                     # RAG 프롬프트
    │       └── agentic/                 # Agentic 프롬프트
    │
    ├── models/                          # 📄 모델 계층
    │   ├── __init__.py
    │   ├── api_schema.py                # API 요청/응답 Pydantic 모델
    │   └── base_models.py               # Document, Message 등 공통 모델
    │
    └── systems/                         # 🧩 시스템 계층
        │
        ├── rag/                         # 📚 RAG 시스템
        │   ├── __init__.py
        │   ├── constants.py             # RAG 상수 정의
        │   ├── rag_system_chain.py      # ⭐ RAG 메인 체인
        │   │
        │   ├── processors/              # 문서 전처리
        │   │   ├── __init__.py
        │   │   ├── document_loader.py   # PDF, TXT 로더
        │   │   └── chunking_strategy.py # 청킹 전략
        │   │
        │   ├── retrievers/              # 검색 전략
        │   │   ├── __init__.py
        │   │   ├── base_retriever.py    # 추상 기본 클래스
        │   │   ├── naive_retriever.py   # 벡터 검색
        │   │   └── hybrid_retriever.py  # 하이브리드 검색
        │   │
        │   └── services/                # 외부 서비스
        │       ├── __init__.py
        │       ├── embedding_service.py # 임베딩 생성
        │       └── vector_store.py      # 벡터 DB 연동
        │
        └── chat/                        # 🤖 Agentic Chat 시스템
            ├── __init__.py
            ├── constants.py             # Agent 상수 정의
            ├── chat_system_chain.py     # ⭐ Chat 메인 체인
            │
            ├── models/                  # Agent 모델
            │   ├── __init__.py
            │   └── function_call_model.py # Function Call 스키마
            │
            ├── prompts/                 # 프롬프트 관리
            │   ├── __init__.py
            │   └── persona_loader.py    # 페르소나 로더
            │
            ├── tools/                   # 도구 구현
            │   ├── __init__.py
            │   ├── tool_registry.py     # 도구 중앙 관리
            │   ├── data_analyzer.py     # 데이터 분석
            │   ├── chart_generator.py   # 차트 생성
            │   ├── report_formatter.py  # 보고서 포맷팅
            │   └── rag_tool.py          # RAG 검색 도구
            │
            └── agents/                  # 에이전트 로직
                ├── __init__.py
                ├── agent_planner.py     # 계획/의사결정
                └── tool_executor.py     # 도구 실행
```

---

## 계층별 역할

### 1. API 계층 (`src/api/`)

외부 HTTP 요청을 처리하고 응답을 반환합니다.

**주요 파일:**
- `router.py`: 모든 라우터를 등록하고 관리
- `chat_endpoints.py`: 채팅 관련 엔드포인트
- `rag_endpoints.py`: RAG 관련 엔드포인트

**엔드포인트:**

| 경로 | 메서드 | 설명 |
|------|--------|------|
| `/api/v1/chat/stream` | POST | 스트리밍 채팅 |
| `/api/v1/chat/message` | POST | 일반 채팅 |
| `/api/v1/chat/prompts` | GET | 프롬프트 목록 |
| `/api/v1/rag/upload` | POST | 문서 업로드 |
| `/api/v1/rag/query` | POST | RAG 질의 |

---

### 2. Config 계층 (`src/config/`)

애플리케이션 전역 설정을 관리합니다.

**주요 파일:**
- `config_model.py`: LLM 모델, API 키, 경로 설정
- `config_prompts.py`: 프롬프트 디렉토리 경로 및 유틸리티
- `config_db.py`: 벡터 DB 연결 정보

**환경 변수:**

```bash
# .env 파일 예시
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=gpt-4o
VLLM_ENABLED=false
EMBEDDING_MODEL=BAAI/bge-m3
```

---

### 3. Core 계층 (`src/core/`)

핵심 서비스와 로직 조립을 담당합니다.

#### graph_factory.py

시스템 선택에 따라 적절한 체인을 생성합니다.

```python
from src.core.graph_factory import GraphFactory

# Chatbot 생성
chain = GraphFactory.create(
    system_type="chatbot",
    session_id="user123",
    prompt_file="path/to/prompt.yaml"
)

# RAG 생성
rag = GraphFactory.create(
    system_type="rag",
    session_id="user123"
)

# Agentic Chat 생성
chat = GraphFactory.create(
    system_type="chat",
    session_id="user123",
    persona="01-agentic-rag-default"
)
```

#### session_manager.py

세션 및 대화 기록을 관리합니다.

```python
from src.core.session_manager import (
    get_session_history,
    clear_session,
    session_exists
)

# 세션 히스토리 가져오기
history = get_session_history("user123")

# 세션 초기화
clear_session("user123")
```

#### llm_service.py

다양한 LLM 백엔드를 통합 관리합니다.

```python
from src.core.llm_service import create_llm, LLMService

# 단순 생성
llm = create_llm(model="gpt-4o", temperature=0.7)

# 서비스 클래스 사용
service = LLMService(model="gpt-4o")
response = await service.invoke("Hello")
```

---

### 4. Models 계층 (`src/models/`)

애플리케이션 전체의 데이터 구조를 정의합니다.

**주요 모델:**

```python
# API 스키마
class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "gpt-4o"
    prompt_file: str
    temperature: float = 0.0

# 기본 모델
class Document(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class Message(BaseModel):
    role: MessageRole
    content: str
```

---

### 5. Systems 계층 (`src/systems/`)

특정 기능에 특화된 독립적인 시스템입니다.

#### RAG 시스템 (`systems/rag/`)

문서 기반 검색 증강 생성 시스템입니다.

**사용 예시:**

```python
from src.systems.rag import RAGSystemChain

# RAG 시스템 생성
rag = RAGSystemChain(
    session_id="user123",
    use_hybrid_search=True,
    use_reranking=False
)

# 문서 인제스트
result = await rag.ingest_document("path/to/document.pdf")

# 질의
response = await rag.query("문서 내용에 대해 설명해주세요")

# 스트리밍 질의
async for chunk in rag.query_stream("질문"):
    print(chunk, end="")
```

**RAG 설정:**

```python
@dataclass
class RAGConfig:
    model: str = "gpt-4o"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    use_hybrid_search: bool = True
    retrieval_k: int = 10
    final_k: int = 5
```

#### Chat 시스템 (`systems/chat/`)

Agentic AI 기반 도구 활용 채팅 시스템입니다.

**사용 예시:**

```python
from src.systems.chat import ChatSystemChain

# Chat 시스템 생성
chat = ChatSystemChain(
    session_id="user123",
    model="gpt-4o",
    tools=["data_analyzer", "chart_generator"]
)

# 채팅
response = await chat.chat("데이터를 분석해주세요")

# 스트리밍 채팅
async for chunk in chat.chat_stream("차트를 만들어주세요"):
    print(chunk, end="")
```

**제공 도구:**

| 도구 | 설명 |
|------|------|
| `data_analyzer` | 데이터 분석 및 통계 계산 |
| `chart_generator` | 차트 이미지 생성 |
| `report_formatter` | 보고서 포맷팅 |
| `rag_tool` | 문서 검색 |

---

## 데이터 흐름

### 1. 채팅 요청 흐름

```
Client → FastAPI /api/v1/chat/stream
           ↓
       chat_endpoints.py
           ↓
       graph_factory.create_chatbot_chain()
           ↓
       ChatPromptTemplate + LLM + StrOutputParser
           ↓
       RunnableWithMessageHistory
           ↓
       chain.astream() → SSE 응답
```

### 2. RAG 문서 업로드 흐름

```
Client → FastAPI /api/v1/rag/upload
           ↓
       rag_endpoints.py
           ↓
       graph_factory.create_rag_system()
           ↓
       RAGSystemChain.ingest_document()
           ↓
       DocumentLoader → Chunker → Retriever
           ↓
       FAISS/BM25 인덱스 생성
```

### 3. RAG 질의 흐름

```
Client → FastAPI /api/v1/rag/query
           ↓
       rag_endpoints.py
           ↓
       RAGSystemChain.query_stream()
           ↓
       Retriever.retrieve() → Context 생성
           ↓
       Prompt + LLM → 스트리밍 응답
```

---

## 실행 방법

### 1. 환경 설정

```bash
# ai-backend 폴더로 이동
cd ai-backend

# 의존성 설치
poetry install

# .env 파일 생성
cp .env.example .env
# API 키 설정
```

### 2. 서버 실행

```bash
# 개발 모드
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 또는 직접 실행
python main.py
```

### 3. API 문서 확인

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 기존 ai_backend와의 차이점

| 항목 | ai_backend (기존) | ai-backend (신규) |
|------|-------------------|-------------------|
| 폴더 구조 | `app/` 기반 | `src/` 기반 |
| 프롬프트 위치 | `prompts/` | `data/prompts/` |
| 팩토리 이름 | `chain_factory.py` | `graph_factory.py` |
| RAG 구현 | 단일 파일 | 모듈식 (processors, retrievers, services) |
| Chat 구현 | 없음 | Agentic AI 기반 |
| 도구 시스템 | 없음 | Tool Registry 패턴 |
| 코드 재사용 | 일부 중복 | 공통 컴포넌트 분리 |

---

## 확장 가이드

### 새로운 Retriever 추가

```python
# src/systems/rag/retrievers/my_retriever.py
from .base_retriever import BaseRetriever

class MyRetriever(BaseRetriever):
    async def retrieve(self, query: str, k: int = None):
        # 검색 로직 구현
        pass

    async def add_documents(self, documents):
        # 문서 추가 로직
        pass
```

### 새로운 Tool 추가

```python
# src/systems/chat/tools/my_tool.py
from .tool_registry import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "새로운 도구"

    def get_definition(self):
        return ToolDefinition(...)

    async def execute(self, **kwargs):
        # 실행 로직
        pass

# 등록
from .tool_registry import register_tool
register_tool(MyTool())
```

---

## 참고 사항

- 모든 비동기 함수는 `async/await` 패턴 사용
- 타입 힌팅 적용
- Pydantic v2 모델 사용
- LangChain/LangGraph 기반

---

## 파일 목록

총 **45+ 파일** 생성:

- API: 4개
- Config: 4개
- Core: 4개
- Models: 3개
- RAG System: 12개
- Chat System: 15개
- 기타: 3개

---

*문서 작성일: 2025-11-18*
*버전: 0.2.0*

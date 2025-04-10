# %%
import os
import sys
import time
import json
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, font, filedialog, messagebox
from datetime import datetime
import uuid
import importlib.util
import inspect
from abc import ABC, abstractmethod
from dotenv import load_dotenv # dotenv 추가

# .env 파일 로드 (Utils 폴더 내 .env 파일 경로 지정)
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)


# 임베딩용
from sentence_transformers import SentenceTransformer

# 벡터 DB
import numpy as np
import faiss

# 옵션: Pinecone
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# 기본 LLM API들
try:
    import google.generativeai as genai  # Gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import groq  # Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# =============================================
# LLM 추상 인터페이스 및 구현체
# =============================================
class LLMService(ABC):
    """LLM 서비스 추상 클래스 - 사용자 정의 LLM 구현시 이 클래스를 상속해야 함"""
    
    @abstractmethod
    def initialize(self, api_key=None):
        """LLM 서비스 초기화"""
        pass
    
    @abstractmethod
    def generate_response(self, messages, tools=None):
        """메시지에 대한 응답 생성"""
        pass
    
    @abstractmethod
    def get_name(self):
        """LLM 서비스 이름 반환"""
        pass


class GeminiService(LLMService):
    """Google Gemini API 서비스"""
    
    def initialize(self, api_key=None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google.generativeai 모듈이 설치되지 않았습니다. pip install google-generativeai로 설치하세요.")
        
        if api_key:
            genai.configure(api_key=api_key)
        self.model = "gemini-pro"
        return True
    
    def generate_response(self, messages, tools=None):
        model = genai.GenerativeModel(self.model)
        
        # 메시지 형식 변환 (Gemini 형식으로)
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Gemini는 시스템 메시지를 직접 지원하지 않음, 사용자 메시지로 변환
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                gemini_messages.append({"role": "model", "parts": ["이해했습니다. 지시사항에 따라 응답하겠습니다."]})
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
        
        # 함수 호출 기능 처리
        if tools:
            # Gemini의 함수 호출 기능은 아직 제한적임
            # 텍스트 응답으로 함수 호출 의도 확인
            try:
                response = model.generate_content(gemini_messages)
                response_text = response.text
                
                # 응답에서 함수 호출 의도 파싱 (간단한 구현)
                if "실행 명령어:" in response_text:
                    lines = response_text.split("\n")
                    for line in lines:
                        if line.startswith("실행 명령어:"):
                            command = line.replace("실행 명령어:", "").strip()
                            return {
                                "content": response_text,
                                "function_call": {
                                    "name": "execute_terminal_command",
                                    "arguments": json.dumps({
                                        "command": command,
                                        "reason": "사용자 요청에 따른 명령어 실행"
                                    })
                                }
                            }
                
                return {"content": response_text}
            except Exception as e:
                return {"content": f"오류 발생: {str(e)}"}
        else:
            try:
                response = model.generate_content(gemini_messages)
                return {"content": response.text}
            except Exception as e:
                return {"content": f"오류 발생: {str(e)}"}
    
    def get_name(self):
        return "Google Gemini"


class GroqService(LLMService):
    """Groq API 서비스"""
    
    def initialize(self, api_key=None):
        if not GROQ_AVAILABLE:
            raise ImportError("groq 모듈이 설치되지 않았습니다. pip install groq로 설치하세요.")
        
        if api_key:
            self.client = groq.Client(api_key=api_key)
        else:
            self.client = groq.Client()
        
        self.model = "llama3-70b-8192"  # 기본 모델
        return True
    
    def generate_response(self, messages, tools=None):
        try:
            if tools:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
            
            return response.choices[0].message.model_dump()
        except Exception as e:
            return {"content": f"오류 발생: {str(e)}"}
    
    def get_name(self):
        return "Groq API"


# =============================================
# 벡터 데이터베이스 추상 인터페이스 및 구현체
# =============================================
class VectorDBService(ABC):
    """벡터 DB 서비스 추상 클래스"""
    
    @abstractmethod
    def initialize(self, embedding_dim, collection_name):
        """벡터 DB 초기화"""
        pass
    
    @abstractmethod
    def add_document(self, document, metadata, doc_id):
        """문서 추가"""
        pass
    
    @abstractmethod
    def search(self, query_vector, limit=5):
        """유사 문서 검색"""
        pass


class FAISSVectorDB(VectorDBService):
    """FAISS 벡터 DB 구현"""
    
    def initialize(self, embedding_dim, collection_name):
        """FAISS 인덱스 초기화"""
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 거리 기반 인덱스
        self.documents = []
        self.metadata = []
        self.ids = []
        self.collection_name = collection_name
        
        # 인덱스 파일 경로
        self.index_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal", "faiss_indices")
        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path = os.path.join(self.index_dir, f"{collection_name}.index")
        self.metadata_path = os.path.join(self.index_dir, f"{collection_name}_metadata.json")
        
        # 기존 인덱스 로드 시도
        self._load_index()
        return True
    
    def _load_index(self):
        """저장된 인덱스 로드"""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                
                # 메타데이터 로드
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.documents = data.get("documents", [])
                        self.metadata = data.get("metadata", [])
                        self.ids = data.get("ids", [])
                
                print(f"FAISS 인덱스 로드됨: {self.collection_name}, 문서 수: {len(self.documents)}")
        except Exception as e:
            print(f"인덱스 로드 오류: {str(e)}")
    
    def _save_index(self):
        """인덱스 저장"""
        try:
            faiss.write_index(self.index, self.index_path)
            
            # 메타데이터 저장
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump({
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "ids": self.ids
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"인덱스 저장 오류: {str(e)}")
    
    def add_document(self, document, metadata, doc_id):
        """문서 추가"""
        if isinstance(document, str):
            return False  # 문자열 대신 벡터를 전달해야 함
        
        # 문서 벡터가 2D 배열이면 1D로 변환
        if len(document.shape) > 1:
            document = document.reshape(1, -1)
        
        # FAISS 인덱스에 추가
        self.index.add(document)
        
        # 문서 정보 저장
        doc_text = metadata.get("text", "")
        self.documents.append(doc_text)
        self.metadata.append(metadata)
        self.ids.append(doc_id)
        
        # 주기적으로 저장 (여기서는 간단히 매번 저장)
        self._save_index()
        return True
    
    def search(self, query_vector, limit=5):
        """유사 문서 검색"""
        if len(self.documents) == 0:
            return {"documents": [], "metadatas": [], "ids": []}
        
        # 차원 확인 및 재구성
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 검색
        distances, indices = self.index.search(query_vector, min(limit, len(self.documents)))
        
        # 결과 구성
        result_docs = []
        result_metadata = []
        result_ids = []
        
        for idx in indices[0]:
            if idx < len(self.documents) and idx >= 0:
                result_docs.append(self.documents[idx])
                result_metadata.append(self.metadata[idx])
                result_ids.append(self.ids[idx])
        
        return {
            "documents": result_docs,
            "metadatas": result_metadata,
            "ids": result_ids
        }


class PineconeVectorDB(VectorDBService):
    """Pinecone 벡터 DB 구현"""
    
    def initialize(self, embedding_dim, collection_name):
        """Pinecone 인덱스 초기화"""
        if not PINECONE_AVAILABLE:
            raise ImportError("pinecone-client 모듈이 설치되지 않았습니다.")
        
        try:
            # API 키 환경 변수에서 가져오기
            api_key = os.environ.get("PINECONE_API_KEY", "")
            if not api_key:
                raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
            
            # Pinecone 초기화
            pinecone.init(api_key=api_key)
            
            # 인덱스 존재 확인 또는 생성
            if collection_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=collection_name,
                    dimension=embedding_dim,
                    metric="cosine"
                )
            
            # 인덱스 연결
            self.index = pinecone.Index(collection_name)
            self.collection_name = collection_name
            return True
        except Exception as e:
            print(f"Pinecone 초기화 오류: {str(e)}")
            return False
    
    def add_document(self, document, metadata, doc_id):
        """문서 추가"""
        try:
            # FAISS와 달리 문자열이 아닌 벡터 형태의 document 필요
            if isinstance(document, np.ndarray):
                document = document.tolist()
            
            # Pinecone에 문서 추가
            self.index.upsert([(doc_id, document, metadata)])
            return True
        except Exception as e:
            print(f"문서 추가 오류: {str(e)}")
            return False
    
    def search(self, query_vector, limit=5):
        """유사 문서 검색"""
        try:
            # NumPy 배열을 리스트로 변환
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            
            # Pinecone 검색
            results = self.index.query(
                vector=query_vector,
                top_k=limit,
                include_metadata=True
            )
            
            # 결과 구성
            documents = []
            metadatas = []
            ids = []
            
            for match in results["matches"]:
                # 메타데이터에서 텍스트 추출
                text = match["metadata"].get("text", "")
                documents.append(text)
                metadatas.append(match["metadata"])
                ids.append(match["id"])
            
            return {
                "documents": documents,
                "metadatas": metadatas,
                "ids": ids
            }
        except Exception as e:
            print(f"검색 오류: {str(e)}")
            return {"documents": [], "metadatas": [], "ids": []}


# =============================================
# Tool 시스템 (사용자 정의 Tool 지원)
# =============================================
class ToolRegistry:
    """Tool 등록 및 관리 클래스"""
    
    def __init__(self):
        self.tools = {}
        self.default_tools = {}
        self._initialize_default_tools()
    
    def _initialize_default_tools(self):
        """기본 Tool 초기화"""
        # 명령어 실행 Tool
        self.default_tools["execute_terminal_command"] = {
            "name": "execute_terminal_command",
            "description": "터미널에서 명령어를 실행합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "실행할 명령어"
                    },
                    "reason": {
                        "type": "string",
                        "description": "이 명령어를 실행하는 이유"
                    }
                },
                "required": ["command", "reason"]
            }
        }
    
    def register_tool(self, tool_name, tool_def, callback_fn):
        """Tool 등록"""
        self.tools[tool_name] = {
            "definition": tool_def,
            "callback": callback_fn
        }
    
    def get_tools_for_llm(self):
        """LLM API에 전달할 Tool 정의 반환"""
        tool_defs = []
        
        # 기본 Tool 추가
        for tool_name, tool_def in self.default_tools.items():
            tool_defs.append(tool_def)
        
        # 사용자 정의 Tool 추가
        for tool_name, tool_info in self.tools.items():
            tool_defs.append(tool_info["definition"])
        
        return tool_defs
    
    def execute_tool(self, tool_name, arguments):
        """Tool 실행"""
        if tool_name in self.tools:
            return self.tools[tool_name]["callback"](arguments)
        else:
            return None


# =============================================
# 메인 터미널 애플리케이션
# =============================================
class AITerminal(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("모듈형 AI 통합 터미널")
        self.geometry("1200x700")
        
        # 세션 ID 생성
        self.session_id = str(uuid.uuid4())
        
        # 임베딩 모델 로드
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Tool 레지스트리
        self.tool_registry = ToolRegistry()
        
        # 메인 프레임 설정
        self.main_frame = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 왼쪽 패널 (CMD 터미널)
        self.terminal_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.terminal_frame, weight=3)
        
        # 오른쪽 패널 (채팅 및 세션)
        self.right_panel = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.main_frame.add(self.right_panel, weight=2)
        
        # 채팅 프레임
        self.chat_frame = ttk.Frame(self.right_panel)
        self.right_panel.add(self.chat_frame, weight=2)
        
        # 세션 프레임
        self.session_frame = ttk.Frame(self.right_panel)
        self.right_panel.add(self.session_frame, weight=1)
        
        # 현재 작업 디렉토리 초기화 (UI 설정 전에 필요)
        self.current_dir = os.getcwd()

        # LLM 서비스 선택 및 초기화 (UI 설정 전에 필요)
        self._setup_llm_service()
        
        # 대화 기록, 명령어 관련 속성 초기화 (UI 설정 전에 필요)
        self.chat_history = []
        self.auto_execute = tk.BooleanVar(value=False)
        self.command_queue = []
        
        # 터미널 UI 설정
        self._setup_terminal_ui()
        
        # 채팅 UI 설정
        self._setup_chat_ui()
        
        # 세션 UI 설정
        self._setup_session_ui()
        
        # 메뉴바 설정
        self._setup_menu()
        
        # 벡터 DB 선택 및 초기화
        self._setup_vector_db()
        
        # 사용자 정의 도구 로드
        self._load_custom_tools()
        
        # 터미널 초기 메시지
        # LLM 서비스가 None일 경우 처리 추가
        llm_name = self.llm_service.get_name() if self.llm_service else "설정 실패"
        self.write_terminal(f"모듈형 AI 통합 터미널에 오신 것을 환영합니다!\n")
        self.write_terminal(f"LLM: {llm_name}, 벡터 DB: {self.vector_db_type}\n")
        self.write_terminal(f"현재 디렉토리: {self.current_dir}\n")
        self.update_prompt()
        
        # 세션 로드
        self.load_sessions()
    
    def _setup_llm_service(self):
        """LLM 서비스 설정 (API 키 확인 및 Fallback 강화)"""
        config_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal")
        config_path = os.path.join(config_dir, "config.json")

        # 기본값 설정
        default_llm_type = "groq"
        self.llm_type = default_llm_type
        self.api_keys = {}

        # 설정 파일 로드
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.llm_type = config.get("llm_type", default_llm_type)
                    self.api_keys = config.get("api_keys", {})
            except Exception as e:
                print(f"설정 파일 로드 오류: {e}")
                self.llm_type = default_llm_type
                self.api_keys = {}

        # 1차: 지정된 LLM 시도
        primary_llm_type = self.llm_type
        primary_api_key = self.api_keys.get(primary_llm_type, os.environ.get(f"{primary_llm_type.upper()}_API_KEY", ""))

        if primary_api_key:
            try:
                self.llm_service = self._get_llm_service(primary_llm_type)
                if self.llm_service.initialize(primary_api_key):
                    print(f"{self.llm_service.get_name()} 초기화 성공.")
                    return # 성공 시 종료
                else:
                     messagebox.showwarning("LLM 초기화 실패", f"{self.llm_service.get_name()} 서비스 초기화에 실패했습니다 (키는 존재).")
            except ImportError as e:
                 messagebox.showwarning("LLM 로드 실패", str(e))
            except Exception as e:
                 messagebox.showwarning("LLM 초기화 오류", f"{primary_llm_type} 초기화 중 오류: {e}")
        else:
             messagebox.showwarning("API 키 없음", f"{primary_llm_type} API 키가 설정 파일 또는 환경변수({primary_llm_type.upper()}_API_KEY)에 없습니다.")

        # 2차: Fallback LLM 시도 (Groq -> Gemini 또는 Gemini -> Groq)
        fallback_llm_type = "gemini" if primary_llm_type == "groq" else "groq"
        fallback_available = GEMINI_AVAILABLE if fallback_llm_type == "gemini" else GROQ_AVAILABLE

        if fallback_available:
             messagebox.showinfo("LLM 변경 시도", f"{primary_llm_type} 설정 실패. {fallback_llm_type}(으)로 전환을 시도합니다.")
             fallback_api_key = self.api_keys.get(fallback_llm_type, os.environ.get(f"{fallback_llm_type.upper()}_API_KEY", ""))

             if fallback_api_key:
                 try:
                     self.llm_service = self._get_llm_service(fallback_llm_type)
                     if self.llm_service.initialize(fallback_api_key):
                         print(f"{self.llm_service.get_name()} 초기화 성공.")
                         self.llm_type = fallback_llm_type # 성공 시 타입 변경
                         self._save_config() # 변경된 타입 저장
                         return # 성공 시 종료
                     else:
                         messagebox.showwarning("LLM 초기화 실패", f"{self.llm_service.get_name()} 서비스 초기화에 실패했습니다 (키는 존재).")
                 except ImportError as e:
                    messagebox.showwarning("LLM 로드 실패", str(e))
                 except Exception as e:
                    messagebox.showwarning("LLM 초기화 오류", f"{fallback_llm_type} 초기화 중 오류: {e}")
             else:
                 messagebox.showwarning("API 키 없음", f"{fallback_llm_type} API 키가 설정 파일 또는 환경변수({fallback_llm_type.upper()}_API_KEY)에 없습니다.")

        # 최종 실패
        messagebox.showerror("LLM 설정 실패", "사용 가능한 LLM 서비스를 초기화할 수 없습니다. API 키 설정을 확인하세요.")
        # AI 기능 비활성화 또는 기본 동작 정의 (옵션)
        # 여기서는 일단 None으로 설정하여 이후 코드에서 확인 가능하게 함
        self.llm_service = None
        self.llm_type = None

    def _get_llm_service(self, llm_type):
        """LLM 서비스 인스턴스 반환"""
        if llm_type == "gemini":
            if not GEMINI_AVAILABLE:
                 raise ImportError("google.generativeai 모듈이 설치되지 않았습니다. pip install google-generativeai로 설치하세요.")
            return GeminiService()
        elif llm_type == "groq":
             if not GROQ_AVAILABLE:
                 raise ImportError("groq 모듈이 설치되지 않았습니다. pip install groq로 설치하세요.")
             return GroqService()
        elif llm_type == "custom":
             # 사용자 정의 LLM 로직은 _on_model_change 에 있으므로 여기서는 기본 핸들링 제거
             # 또는 _on_model_change 와 통합 필요
             raise ValueError("사용자 정의 LLM은 메뉴를 통해 설정해야 합니다.")
        else:
            # 지원하지 않는 타입 또는 설정 오류
             raise ValueError(f"지원하지 않는 LLM 타입입니다: {llm_type}")

    def _setup_vector_db(self):
        """벡터 DB 설정"""
        # 기본 벡터 DB 타입
        self.vector_db_type = "faiss"  # 기본값
        
        # 사용자 설정 파일 확인
        config_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal")
        config_path = os.path.join(config_dir, "config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.vector_db_type = config.get("vector_db_type", "faiss")
            except Exception:
                pass
        
        # 벡터 DB 초기화
        if self.vector_db_type == "pinecone" and PINECONE_AVAILABLE:
            # Pinecone 벡터 DB 초기화
            self.terminal_vector_db = PineconeVectorDB()
            self.chat_vector_db = PineconeVectorDB()
        else:
            # 기본 FAISS 벡터 DB 초기화
            self.terminal_vector_db = FAISSVectorDB()
            self.chat_vector_db = FAISSVectorDB()
        
        # 컬렉션 초기화
        self.terminal_vector_db.initialize(self.embedding_dim, f"terminal_logs_{self.session_id}")
        self.chat_vector_db.initialize(self.embedding_dim, f"chat_history_{self.session_id}")
    
    def _load_custom_tools(self):
        """사용자 정의 도구 로드"""
        # 도구 디렉토리
        tools_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal", "tools")
        os.makedirs(tools_dir, exist_ok=True)
        
        # 도구 파일 검색
        if os.path.exists(tools_dir):
            for file in os.listdir(tools_dir):
                if file.endswith(".py"):
                    try:
                        # 파이썬 모듈 동적 로드
                        module_path = os.path.join(tools_dir, file)
                        module_name = os.path.splitext(file)[0]
                        
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # 도구 등록 함수 찾기
                            if hasattr(module, "register_tools"):
                                module.register_tools(self.tool_registry)
                                print(f"도구 모듈 로드됨: {module_name}")
                    except Exception as e:
                        print(f"도구 로드 오류 ({file}): {str(e)}")
    
    def _setup_terminal_ui(self):
        """터미널 UI 설정"""
        terminal_font = font.Font(family="Consolas", size=10)
        
        # 터미널 출력 영역
        self.terminal_output = scrolledtext.ScrolledText(self.terminal_frame, wrap=tk.WORD)
        self.terminal_output.pack(expand=True, fill="both", padx=5, pady=5)
        self.terminal_output.configure(font=terminal_font)
        
        # 입력 프레임
        input_frame = ttk.Frame(self.terminal_frame)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # 프롬프트 레이블
        self.prompt_label = ttk.Label(input_frame, text=f"{self.current_dir}>", font=terminal_font)
        self.prompt_label.pack(side="left")
        
        # 명령어 입력 필드
        self.command_entry = ttk.Entry(input_frame, font=terminal_font)
        self.command_entry.pack(side="left", expand=True, fill="x")
        self.command_entry.bind("<Return>", self.execute_command)
        self.command_entry.bind("<Up>", self.previous_command)
        self.command_entry.bind("<Down>", self.next_command)
        self.command_entry.focus_set()
        
        # 명령어 기록
        self.command_history = []
        self.history_index = 0
    
    def _setup_chat_ui(self):
        """채팅 UI 설정"""
        chat_font = font.Font(family="Segoe UI", size=10)
        
        # 채팅 출력 영역
        chat_label = ttk.Label(self.chat_frame, text="AI 어시스턴트 채팅", font=font.Font(size=12, weight="bold"))
        chat_label.pack(anchor="w", padx=5, pady=5)
        
        self.chat_output = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD)
        self.chat_output.pack(expand=True, fill="both", padx=5, pady=5)
        self.chat_output.configure(font=chat_font)
        self.chat_output.tag_configure("user", foreground="blue")
        self.chat_output.tag_configure("assistant", foreground="green")
        self.chat_output.tag_configure("system", foreground="gray")
        self.chat_output.tag_configure("tool", foreground="purple")
        
        # 입력 프레임
        chat_input_frame = ttk.Frame(self.chat_frame)
        chat_input_frame.pack(fill="x", padx=5, pady=5)
        
        # LLM 모델 선택
        model_frame = ttk.Frame(chat_input_frame)
        model_frame.pack(side="top", fill="x", pady=2)
        
        ttk.Label(model_frame, text="LLM:").pack(side="left")
        self.model_var = tk.StringVar(value=self.llm_type)
        model_combo = ttk.Combobox(
            model_frame, 
            textvariable=self.model_var,
            values=["groq", "gemini", "custom"],
            width=10,
            state="readonly"
        )
        model_combo.pack(side="left", padx=5)
        model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        
        # 자동 실행 체크박스
        auto_execute_check = ttk.Checkbutton(
            chat_input_frame, 
            text="명령어 자동 실행", 
            variable=self.auto_execute
        )
        auto_execute_check.pack(side="left")
        
        # 채팅 입력 필드
        self.chat_entry = ttk.Entry(chat_input_frame, font=chat_font)
        self.chat_entry.pack(side="left", expand=True, fill="x")
        self.chat_entry.bind("<Return>", self.send_message)
        
        # 전송 버튼
        send_button = ttk.Button(chat_input_frame, text="전송", command=self.send_message)
        send_button.pack(side="right")
    
    def _on_model_change(self, event):
        """LLM 모델 변경 처리"""
        new_model = self.model_var.get()
        
        if new_model == "custom":
            # 사용자 정의 모델 파일 선택
            file_path = filedialog.askopenfilename(
                title="사용자 정의 LLM 모듈 선택",
                filetypes=[("Python 파일", "*.py")],
                initialdir=os.path.join(os.path.expanduser("~"), ".ai_terminal", "llm_modules")
            )
            
            if file_path:
                try:
                    # 모듈 동적 로드
                    module_name = os.path.splitext(os.path.basename(file_path))[0]
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # LLMService 클래스 확인
                        llm_class = None
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and issubclass(obj, LLMService) and 
                                obj is not LLMService):
                                llm_class = obj
                                break
                        
                        if llm_class:
                            # 인스턴스 생성 및 초기화
                            self.llm_service = llm_class()
                            api_key = self.api_keys.get("custom")
                            self.llm_service.initialize(api_key)
                            self.llm_type = "custom"
                            
                            # 설정 저장
                            self._save_config()
                            
                            messagebox.showinfo("LLM 변경", f"사용자 정의 LLM으로 변경되었습니다: {self.llm_service.get_name()}")
                        else:
                            messagebox.showerror("오류", "선택한 파일에서 LLMService 상속 클래스를 찾을 수 없습니다.")
                            self.model_var.set(self.llm_type)  # 이전 값으로 복원
                except Exception as e:
                    messagebox.showerror("오류", f"사용자 정의 LLM 로드 중 오류 발생: {str(e)}")
                    self.model_var.set(self.llm_type)  # 이전 값으로 복원
            else:
                self.model_var.set(self.llm_type)  # 취소 시 이전 값으로 복원
        else:
            # 기본 제공 모델 변경
            try:
                self.llm_service = self._get_llm_service(new_model)
                api_key = self.api_keys.get(new_model)
                self.llm_service.initialize(api_key)
                self.llm_type = new_model
                
                # 설정 저장
                self._save_config()
                
                messagebox.showinfo("LLM 변경", f"LLM이 변경되었습니다: {self.llm_service.get_name()}")
            except Exception as e:
                messagebox.showerror("오류", f"LLM 변경 중 오류 발생: {str(e)}")
                self.model_var.set(self.llm_type)  # 이전 값으로 복원
    
    def _save_config(self):
        """설정 저장"""
        config_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "config.json")
        
        config = {
            "llm_type": self.llm_type,
            "vector_db_type": self.vector_db_type,
            "api_keys": self.api_keys
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def _setup_session_ui(self):
        """세션 UI 설정"""
        # 세션 레이블
        session_label = ttk.Label(self.session_frame, text="저장된 세션", font=font.Font(size=12, weight="bold"))
        session_label.pack(anchor="w", padx=5, pady=5)
        
        # 세션 리스트
        session_frame = ttk.Frame(self.session_frame)
        session_frame.pack(expand=True, fill="both", padx=5, pady=5)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(session_frame)
        scrollbar.pack(side="right", fill="y")
        
        # 세션 리스트박스
        self.session_listbox = tk.Listbox(session_frame, yscrollcommand=scrollbar.set)
        self.session_listbox.pack(side="left", expand=True, fill="both")
        scrollbar.config(command=self.session_listbox.yview)
        self.session_listbox.bind("<Double-1>", self.load_selected_session)
        
        # 버튼 프레임
        button_frame = ttk.Frame(self.session_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        # 새 세션 버튼
        new_session_button = ttk.Button(button_frame, text="새 세션", command=self.new_session)
        new_session_button.pack(side="left", padx=2)
        
        # 세션 저장 버튼
        save_session_button = ttk.Button(button_frame, text="세션 저장", command=self.save_current_session)
        save_session_button.pack(side="left", padx=2)
        
        # 세션 삭제 버튼
        delete_session_button = ttk.Button(button_frame, text="세션 삭제", command=self.delete_selected_session)
        delete_session_button.pack(side="left", padx=2)
    
    def _setup_menu(self):
        """메뉴바 설정"""
        menu_bar = tk.Menu(self)
        
        # 파일 메뉴
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="새 터미널", command=self.new_session)
        file_menu.add_command(label="세션 저장", command=self.save_current_session)
        file_menu.add_command(label="세션 열기", command=self.load_sessions)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.quit)
        menu_bar.add_cascade(label="파일", menu=file_menu)
        
        # 편집 메뉴
        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="복사", command=lambda: self.focus_get().event_generate("<<Copy>>") if self.focus_get() else None)
        edit_menu.add_command(label="붙여넣기", command=lambda: self.focus_get().event_generate("<<Paste>>") if self.focus_get() else None)
        edit_menu.add_command(label="잘라내기", command=lambda: self.focus_get().event_generate("<<Cut>>") if self.focus_get() else None)
        menu_bar.add_cascade(label="편집", menu=edit_menu)
        
        # 설정 메뉴
        settings_menu = tk.Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="API 키 설정", command=self.set_api_key)
        
        # 벡터 DB 서브메뉴
        vector_db_menu = tk.Menu(settings_menu, tearoff=0)
        vector_db_menu.add_command(label="FAISS", command=lambda: self.set_vector_db("faiss"))
        if PINECONE_AVAILABLE:
            vector_db_menu.add_command(label="Pinecone", command=lambda: self.set_vector_db("pinecone"))
        settings_menu.add_cascade(label="벡터 DB 선택", menu=vector_db_menu)
        
        # 도구 관리
        settings_menu.add_command(label="도구 관리", command=self.manage_tools)
        
        menu_bar.add_cascade(label="설정", menu=settings_menu)
        
        self.config(menu=menu_bar)
    
    def write_terminal(self, text):
        """터미널에 텍스트 출력 및 벡터 DB에 저장"""
        self.terminal_output.configure(state="normal")
        self.terminal_output.insert(tk.END, text)
        self.terminal_output.see(tk.END)
        self.terminal_output.configure(state="disabled")
        
        # 벡터 DB에 저장 (비동기로 처리)
        threading.Thread(target=self._save_to_vector_db, args=(text,)).start()
    
    def _save_to_vector_db(self, text):
        """텍스트를 벡터 DB에 저장"""
        try:
            # 공백이나 너무 짧은 텍스트는 저장하지 않음
            if len(text.strip()) < 3:
                return
            
            # 임베딩 생성
            embedding = self.embedding_model.encode([text])[0]  # 첫 번째 벡터 사용
            
            # 메타데이터 구성
            metadata = {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }
            
            # 벡터 DB에 저장
            doc_id = str(uuid.uuid4())
            self.terminal_vector_db.add_document(embedding, metadata, doc_id)
        except Exception as e:
            print(f"벡터 DB 저장 오류: {str(e)}")
    
    def update_prompt(self):
        """프롬프트 업데이트"""
        self.current_dir = os.getcwd()
        self.prompt_label.config(text=f"{self.current_dir}>")
    
    def execute_command(self, event=None):
        """명령어 실행"""
        command = self.command_entry.get()
        self.command_entry.delete(0, tk.END)
        
        if not command:
            return
        
        # 명령어 기록에 추가
        self.command_history.append(command)
        self.history_index = len(self.command_history)
        
        # 명령어 출력
        self.write_terminal(f"{command}\n")
        
        # 내장 명령어 처리
        if command.lower() == "exit":
            self.quit()
            return
        elif command.lower() in ["cls", "clear"]:
            self.terminal_output.configure(state="normal")
            self.terminal_output.delete(1.0, tk.END)
            self.terminal_output.configure(state="disabled")
            return
        elif command.lower().startswith("cd "):
            self._change_directory(command[3:].strip())
            return
            
        # 외부 명령어 실행
        self._run_command(command)
    
    def _change_directory(self, path):
        """디렉토리 변경"""
        try:
            # 상대 경로 처리
            if not os.path.isabs(path):
                path = os.path.join(self.current_dir, path)
            
            # 경로 정규화
            path = os.path.normpath(path)
            
            if os.path.exists(path) and os.path.isdir(path):
                os.chdir(path)
                self.update_prompt()
            else:
                self.write_terminal("지정된 경로를 찾을 수 없습니다.\n")
        except Exception as e:
            self.write_terminal(f"오류: {str(e)}\n")
    
    def _run_command(self, command):
        """명령어 실행 (비동기)"""
        threading.Thread(target=self._execute_command_thread, args=(command,)).start()
    
    def _execute_command_thread(self, command):
        """명령어 실행 스레드"""
        try:
            # 환경 변수 복사 (conda, ffmpeg 등 경로가 포함된)
            env = os.environ.copy()
            
            # 명령어 실행
            if sys.platform == 'win32':
                # 윈도우 환경에서 실행
                process = subprocess.Popen(
                    ["cmd.exe", "/c", command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    cwd=self.current_dir,
                    env=env,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
            else:
                # 유닉스 환경에서 실행
                process = subprocess.Popen(
                    ["/bin/bash", "-c", command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    cwd=self.current_dir,
                    env=env,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
            
            # 출력 실시간 읽기
            if process.stdout:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        self.write_terminal(output)
            
            # 에러 출력 읽기
            if process.stderr:
                for error in process.stderr:
                    self.write_terminal(error)
            
            # 프로세스 종료 코드
            return_code = process.poll()
            if return_code != 0:
                self.write_terminal(f"명령어가 코드 {return_code}로 종료되었습니다.\n")
                
        except Exception as e:
            self.write_terminal(f"명령어 실행 오류: {str(e)}\n")
    
    def previous_command(self, event=None):
        """이전 명령어 가져오기"""
        if not self.command_history:
            return "break"
        
        if self.history_index > 0:
            self.history_index -= 1
            self.command_entry.delete(0, tk.END)
            self.command_entry.insert(0, self.command_history[self.history_index])
        return "break"
    
    def next_command(self, event=None):
        """다음 명령어 가져오기"""
        if not self.command_history:
            return "break"
        
        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.command_entry.delete(0, tk.END)
            self.command_entry.insert(0, self.command_history[self.history_index])
        elif self.history_index == len(self.command_history) - 1:
            self.history_index += 1
            self.command_entry.delete(0, tk.END)
        return "break"
    
    def send_message(self, event=None):
        """메시지 전송"""
        message = self.chat_entry.get()
        if not message:
            return
        
        self.chat_entry.delete(0, tk.END)
        
        # 사용자 메시지 표시
        self.chat_output.configure(state="normal")
        self.chat_output.insert(tk.END, f"사용자: {message}\n", "user")
        self.chat_output.see(tk.END)
        self.chat_output.configure(state="disabled")
        
        # 대화 기록에 추가
        self.chat_history.append({"role": "user", "content": message})
        
        # 채팅 임베딩 및 저장
        threading.Thread(target=self._save_chat_to_vector_db, args=(message, "user")).start()
        
        # 비동기로 응답 생성
        threading.Thread(target=self._generate_response, args=(message,)).start()
    
    def _save_chat_to_vector_db(self, message, role):
        """채팅 메시지를 벡터 DB에 저장"""
        try:
            # 임베딩 생성
            embedding = self.embedding_model.encode([message])[0]
            
            # 메타데이터 구성
            metadata = {
                "text": message,
                "role": role,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }
            
            # 벡터 DB에 저장
            doc_id = str(uuid.uuid4())
            self.chat_vector_db.add_document(embedding, metadata, doc_id)
        except Exception as e:
            print(f"채팅 벡터 DB 저장 오류: {str(e)}")
    
    def _generate_response(self, user_message):
        """AI 응답 생성"""
        try:
            # 시스템 메시지 표시
            self.chat_output.configure(state="normal")
            self.chat_output.insert(tk.END, "AI 어시스턴트: 생각 중...\n", "system")
            self.chat_output.see(tk.END)
            self.chat_output.configure(state="disabled")
            
            # 관련 터미널 컨텍스트 검색 (RAG)
            context = self._retrieve_context(user_message)
            
            # 시스템 메시지 구성
            system_message = (
                "당신은 터미널 인터페이스에 통합된 AI 어시스턴트입니다. "
                "사용자가 터미널 명령어에 대해 문의하거나 도움을 요청하면 도와주세요. "
                "필요한 경우 사용자를 대신해 터미널 명령어를 직접 실행할 수 있습니다. "
                "다음은 최근 터미널 로그 컨텍스트입니다:\n\n" + context
            )
            
            # 메시지 준비
            messages = [
                {"role": "system", "content": system_message}
            ]
            
            # 대화 기록 추가 (최대 10개 메시지)
            history_to_include = self.chat_history[-10:] if len(self.chat_history) > 10 else self.chat_history
            messages.extend(history_to_include)
            
            # 도구 정의
            tools = self.tool_registry.get_tools_for_llm()
            
            # LLM API 호출
            response = self.llm_service.generate_response(messages, tools)
            
            # 함수 호출이 포함된 경우
            if isinstance(response, dict) and "function_call" in response:
                function_call = response.get("function_call")
                if isinstance(function_call, dict) and "name" in function_call:
                    function_name = function_call["name"]
                    
                    try:
                        function_args = json.loads(function_call["arguments"])
                    except:
                        function_args = function_call["arguments"]
                    
                    # 터미널 명령어 실행 함수 처리
                    if function_name == "execute_terminal_command":
                        command = function_args.get("command", "")
                        reason = function_args.get("reason", "이유가 지정되지 않았습니다.")
                        
                        # 응답 표시
                        self.chat_output.configure(state="normal")
                        self.chat_output.delete("end-2l", "end-1l")  # "생각 중..." 메시지 삭제
                        self.chat_output.insert(tk.END, f"AI 어시스턴트: 다음 명령어를 실행하려고 합니다:\n", "assistant")
                        self.chat_output.insert(tk.END, f"$ {command}\n", "tool")
                        self.chat_output.insert(tk.END, f"이유: {reason}\n", "assistant")
                        if isinstance(response, dict) and response.get("content"):
                            self.chat_output.insert(tk.END, f"{response['content']}\n", "assistant")
                        self.chat_output.see(tk.END)
                        self.chat_output.configure(state="disabled")
                        
                        # 채팅 기록에 추가
                        response_text = f"다음 명령어를 실행하려고 합니다: {command}\n이유: {reason}"
                        if isinstance(response, dict) and response.get("content"):
                            response_text += f"\n{response['content']}"
                        
                        self.chat_history.append({"role": "assistant", "content": response_text})
                        
                        # 채팅 벡터 DB에 저장
                        self._save_chat_to_vector_db(response_text, "assistant")
                        
                        # 자동 실행 모드인 경우 바로 실행
                        if self.auto_execute.get():
                            self.command_entry.insert(0, command)
                            self.execute_command()
                        else:
                            # 실행 확인 메시지
                            if messagebox.askyesno("명령어 실행", f"다음 명령어를 실행하시겠습니까?\n\n{command}"):
                                self.command_entry.insert(0, command)
                                self.execute_command()
                    
                    # 다른 사용자 정의 도구 실행
                    else:
                        result = self.tool_registry.execute_tool(function_name, function_args)
                        
                        # 도구 실행 결과 표시
                        self.chat_output.configure(state="normal")
                        self.chat_output.delete("end-2l", "end-1l")  # "생각 중..." 메시지 삭제
                        
                        if isinstance(response, dict) and response.get("content"):
                            self.chat_output.insert(tk.END, f"AI 어시스턴트: {response['content']}\n", "assistant")
                        
                        self.chat_output.insert(tk.END, f"도구 실행: {function_name}\n", "system")
                        if result:
                            self.chat_output.insert(tk.END, f"결과: {result}\n", "tool")
                        
                        self.chat_output.see(tk.END)
                        self.chat_output.configure(state="disabled")
                
            elif isinstance(response, dict):
                # 일반 텍스트 응답
                response_text = response.get("content", "응답을 생성할 수 없습니다.")
                
                # 응답 표시
                self.chat_output.configure(state="normal")
                self.chat_output.delete("end-2l", "end-1l")  # "생각 중..." 메시지 삭제
                self.chat_output.insert(tk.END, f"AI 어시스턴트: {response_text}\n", "assistant")
                self.chat_output.see(tk.END)
                self.chat_output.configure(state="disabled")
                
                # 대화 기록에 추가
                self.chat_history.append({"role": "assistant", "content": response_text})
                
                # 채팅 벡터 DB에 저장
                self._save_chat_to_vector_db(response_text, "assistant")
        
        except Exception as e:
            # 오류 표시
            self.chat_output.configure(state="normal")
            self.chat_output.delete("end-2l", "end-1l")  # "생각 중..." 메시지 삭제
            self.chat_output.insert(tk.END, f"AI 어시스턴트: 오류가 발생했습니다. {str(e)}\n", "system")
            self.chat_output.see(tk.END)
            self.chat_output.configure(state="disabled")
    
    def _retrieve_context(self, query, n_results=5):
        """RAG: 쿼리와 관련된 터미널 컨텍스트 검색"""
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_model.encode([query])[0]
            
            # 터미널 로그에서 관련 컨텍스트 검색
            results = self.terminal_vector_db.search(query_embedding, n_results)
            
            # 검색 결과가 있는 경우
            if results and len(results["documents"]) > 0:
                # 검색된 문서 결합
                context = "\n".join(results["documents"])
                return context
            else:
                return "관련 터미널 컨텍스트가 없습니다."
                
        except Exception as e:
            print(f"컨텍스트 검색 오류: {str(e)}")
            return "컨텍스트 검색 중 오류가 발생했습니다."
    
    def set_api_key(self):
        """API 키 설정"""
        # API 키 설정 창
        api_key_window = tk.Toplevel(self)
        api_key_window.title("API 키 설정")
        api_key_window.geometry("400x200")
        api_key_window.resizable(False, False)
        api_key_window.transient(self)
        api_key_window.grab_set()
        
        # API 키 입력 프레임
        frame = ttk.Frame(api_key_window, padding=10)
        frame.pack(fill="both", expand=True)
        
        # Groq API 키
        ttk.Label(frame, text="Groq API 키:").grid(row=0, column=0, sticky="w", pady=5)
        groq_api_key = ttk.Entry(frame, width=40, show="*")
        groq_api_key.grid(row=0, column=1, pady=5, padx=5)
        groq_api_key.insert(0, self.api_keys.get("groq", ""))
        
        # Gemini API 키
        ttk.Label(frame, text="Google API 키:").grid(row=1, column=0, sticky="w", pady=5)
        gemini_api_key = ttk.Entry(frame, width=40, show="*")
        gemini_api_key.grid(row=1, column=1, pady=5, padx=5)
        gemini_api_key.insert(0, self.api_keys.get("gemini", ""))
        
        # Pinecone API 키 (선택적)
        if PINECONE_AVAILABLE:
            ttk.Label(frame, text="Pinecone API 키:").grid(row=2, column=0, sticky="w", pady=5)
            pinecone_api_key = ttk.Entry(frame, width=40, show="*")
            pinecone_api_key.grid(row=2, column=1, pady=5, padx=5)
            pinecone_api_key.insert(0, self.api_keys.get("pinecone", ""))
        
        # 사용자 정의 모듈 API 키
        ttk.Label(frame, text="사용자 정의 API 키:").grid(row=3, column=0, sticky="w", pady=5)
        custom_api_key = ttk.Entry(frame, width=40, show="*")
        custom_api_key.grid(row=3, column=1, pady=5, padx=5)
        custom_api_key.insert(0, self.api_keys.get("custom", ""))
        
        # 버튼 프레임
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        # 저장 버튼
        def save_api_keys():
            self.api_keys = {
                "groq": groq_api_key.get(),
                "gemini": gemini_api_key.get(),
                "custom": custom_api_key.get()
            }
            
            if PINECONE_AVAILABLE:
                self.api_keys["pinecone"] = pinecone_api_key.get()
            
            # 현재 LLM 서비스 다시 초기화
            if self.llm_type in self.api_keys:
                self.llm_service.initialize(self.api_keys[self.llm_type])
            
            # 설정 저장
            self._save_config()
            
            messagebox.showinfo("API 키 설정", "API 키가 저장되었습니다.")
            api_key_window.destroy()
        
        ttk.Button(button_frame, text="저장", command=save_api_keys).pack(side="left", padx=5)
        ttk.Button(button_frame, text="취소", command=api_key_window.destroy).pack(side="left", padx=5)
    
    def set_vector_db(self, vector_db_type):
        """벡터 DB 유형 설정"""
        if vector_db_type == "pinecone" and not PINECONE_AVAILABLE:
            messagebox.showerror("오류", "Pinecone 라이브러리가 설치되지 않았습니다. pip install pinecone-client로 설치하세요.")
            return
        
        # 현재 다른 유형이면 변경
        if vector_db_type != self.vector_db_type:
            if messagebox.askyesno("벡터 DB 변경", f"벡터 DB를 {vector_db_type}로 변경하시겠습니까? 현재 세션이 재시작됩니다."):
                self.vector_db_type = vector_db_type
                
                # 설정 저장
                self._save_config()
                
                # 새 세션 시작
                self.new_session()
    
    def manage_tools(self):
        """도구 관리"""
        # 도구 관리 창
        tool_window = tk.Toplevel(self)
        tool_window.title("도구 관리")
        tool_window.geometry("500x400")
        tool_window.transient(self)
        tool_window.grab_set()
        
        # 도구 목록 프레임
        frame = ttk.Frame(tool_window, padding=10)
        frame.pack(fill="both", expand=True)
        
        # 라벨
        ttk.Label(frame, text="설치된 도구", font=font.Font(size=12, weight="bold")).pack(anchor="w")
        
        # 도구 목록 표시
        tool_list_frame = ttk.Frame(frame)
        tool_list_frame.pack(fill="both", expand=True, pady=10)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(tool_list_frame)
        scrollbar.pack(side="right", fill="y")
        
        # 리스트박스
        tool_listbox = tk.Listbox(tool_list_frame, yscrollcommand=scrollbar.set)
        tool_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=tool_listbox.yview)
        
        # 기본 도구 추가
        for tool_name in self.tool_registry.default_tools:
            tool_listbox.insert(tk.END, f"{tool_name} (기본)")
        
        # 사용자 정의 도구 추가
        for tool_name in self.tool_registry.tools:
            tool_listbox.insert(tk.END, tool_name)
        
        # 버튼 프레임
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=10)
        
        # 도구 추가 버튼
        def add_tool():
            file_path = filedialog.askopenfilename(
                title="도구 모듈 선택",
                filetypes=[("Python 파일", "*.py")],
                initialdir=os.path.join(os.path.expanduser("~"), ".ai_terminal", "tools")
            )
            
            if file_path:
                try:
                    # 도구 디렉토리에 복사
                    tools_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal", "tools")
                    os.makedirs(tools_dir, exist_ok=True)
                    
                    import shutil
                    dest_path = os.path.join(tools_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)
                    
                    # 도구 로드
                    module_name = os.path.splitext(os.path.basename(file_path))[0]
                    spec = importlib.util.spec_from_file_location(module_name, dest_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # 도구 등록
                        if hasattr(module, "register_tools"):
                            module.register_tools(self.tool_registry)
                            
                            # 리스트 업데이트
                            tool_listbox.delete(0, tk.END)
                            for tool_name in self.tool_registry.default_tools:
                                tool_listbox.insert(tk.END, f"{tool_name} (기본)")
                            for tool_name in self.tool_registry.tools:
                                tool_listbox.insert(tk.END, tool_name)
                            
                            messagebox.showinfo("도구 추가", "도구가 성공적으로 추가되었습니다.")
                        else:
                            messagebox.showerror("오류", "선택한 파일에 register_tools 함수가 없습니다.")
                except Exception as e:
                    messagebox.showerror("오류", f"도구 추가 중 오류 발생: {str(e)}")
        
        ttk.Button(button_frame, text="도구 추가", command=add_tool).pack(side="left", padx=5)
        ttk.Button(button_frame, text="닫기", command=tool_window.destroy).pack(side="right", padx=5)
    
    def new_session(self):
        """새 세션 생성"""
        # 현재 세션 저장 확인
        if messagebox.askyesno("세션 저장", "현재 세션을 저장하시겠습니까?"):
            self.save_current_session()
        
        # 새 세션 ID 생성
        self.session_id = str(uuid.uuid4())
        
        # 출력 영역 초기화
        self.terminal_output.configure(state="normal")
        self.terminal_output.delete(1.0, tk.END)
        self.terminal_output.configure(state="disabled")
        
        self.chat_output.configure(state="normal")
        self.chat_output.delete(1.0, tk.END)
        self.chat_output.configure(state="disabled")
        
        # 대화 기록 초기화
        self.chat_history = []
        
        # 명령어 기록 초기화
        self.command_history = []
        self.history_index = 0
        
        # 벡터 DB 컬렉션 다시 초기화
        self.terminal_vector_db.initialize(self.embedding_dim, f"terminal_logs_{self.session_id}")
        self.chat_vector_db.initialize(self.embedding_dim, f"chat_history_{self.session_id}")
        
        # 터미널 초기 메시지
        self.write_terminal(f"새 세션이 시작되었습니다. 세션 ID: {self.session_id}\n")
        self.write_terminal(f"LLM: {self.llm_service.get_name()}, 벡터 DB: {self.vector_db_type}\n")
        self.write_terminal(f"현재 디렉토리: {self.current_dir}\n")
        self.update_prompt()
    
    def save_current_session(self):
        """현재 세션 저장"""
        try:
            # 세션 데이터 구조화
            session_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "name": f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "terminal_content": self.terminal_output.get(1.0, tk.END),
                "chat_history": self.chat_history,
                "command_history": self.command_history,
                "current_dir": self.current_dir,
                "llm_type": self.llm_type,
                "vector_db_type": self.vector_db_type
            }
            
            # 세션 저장 디렉토리 확인
            session_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal", "sessions")
            os.makedirs(session_dir, exist_ok=True)
            
            # 세션 파일 저장
            session_file = os.path.join(session_dir, f"{self.session_id}.json")
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("세션 저장", "세션이 성공적으로 저장되었습니다.")
            
            # 세션 목록 새로고침
            self.load_sessions()
            
        except Exception as e:
            messagebox.showerror("저장 오류", f"세션 저장 중 오류 발생: {str(e)}")
    
    def load_sessions(self):
        """저장된 세션 목록 로드"""
        try:
            # 세션 디렉토리 확인
            session_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal", "sessions")
            if not os.path.exists(session_dir):
                return
            
            # 세션 목록 지우기
            self.session_listbox.delete(0, tk.END)
            
            # 세션 파일 읽기
            session_files = [f for f in os.listdir(session_dir) if f.endswith(".json")]
            
            for file in session_files:
                try:
                    with open(os.path.join(session_dir, file), "r", encoding="utf-8") as f:
                        session_data = json.load(f)
                        # 세션 이름 표시 (있으면 이름, 없으면 타임스탬프)
                        session_name = session_data.get("name", session_data.get("timestamp", file))
                        self.session_listbox.insert(tk.END, session_name)
                        # 세션 ID를 항목 데이터로 저장
                        self.session_listbox.itemconfig(tk.END, session_id=session_data["session_id"])
                except Exception as e:
                    print(f"세션 파일 '{file}' 로드 중 오류: {str(e)}")
            
        except Exception as e:
            messagebox.showerror("로드 오류", f"세션 목록 로드 중 오류 발생: {str(e)}")
    
    def load_selected_session(self, event=None):
        """선택한 세션 로드"""
        try:
            selection = self.session_listbox.curselection()
            if not selection:
                return
            
            # 선택한 항목의 세션 ID 가져오기
            index = selection[0]
            session_id = self.session_listbox.itemcget(index, "session_id")
            
            # 세션 파일 경로
            session_file = os.path.join(
                os.path.expanduser("~"), 
                ".ai_terminal", 
                "sessions", 
                f"{session_id}.json"
            )
            
            # 세션 데이터 로드
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            
            # 세션 정보 복원
            self.session_id = session_id
            
            # 터미널 컨텐츠 복원
            self.terminal_output.configure(state="normal")
            self.terminal_output.delete(1.0, tk.END)
            self.terminal_output.insert(tk.END, session_data["terminal_content"])
            self.terminal_output.see(tk.END)
            self.terminal_output.configure(state="disabled")
            
            # 채팅 기록 복원
            self.chat_history = session_data.get("chat_history", [])
            
            # 채팅 출력 복원
            self.chat_output.configure(state="normal")
            self.chat_output.delete(1.0, tk.END)
            for msg in self.chat_history:
                if msg["role"] == "user":
                    self.chat_output.insert(tk.END, f"사용자: {msg['content']}\n", "user")
                elif msg["role"] == "assistant":
                    self.chat_output.insert(tk.END, f"AI 어시스턴트: {msg['content']}\n", "assistant")
            self.chat_output.see(tk.END)
            self.chat_output.configure(state="disabled")
            
            # 명령어 기록 복원
            self.command_history = session_data.get("command_history", [])
            self.history_index = len(self.command_history)
            
            # 작업 디렉토리 복원
            current_dir = session_data.get("current_dir", os.getcwd())
            if os.path.exists(current_dir) and os.path.isdir(current_dir):
                os.chdir(current_dir)
                self.current_dir = current_dir
                self.update_prompt()
            
            # LLM 유형 복원 (선택적)
            if "llm_type" in session_data:
                self.llm_type = session_data["llm_type"]
                self.model_var.set(self.llm_type)
                self.llm_service = self._get_llm_service(self.llm_type)
                api_key = self.api_keys.get(self.llm_type)
                self.llm_service.initialize(api_key)
            
            # 벡터 DB 유형 복원 (선택적)
            if "vector_db_type" in session_data:
                self.vector_db_type = session_data["vector_db_type"]
            
            # 벡터 DB 컬렉션 다시 초기화
            self.terminal_vector_db.initialize(self.embedding_dim, f"terminal_logs_{self.session_id}")
            self.chat_vector_db.initialize(self.embedding_dim, f"chat_history_{self.session_id}")
            
            # 세션 로드 메시지
            self.write_terminal(f"\n세션 '{session_data.get('name')}' 로드됨.\n")
            
        except Exception as e:
            messagebox.showerror("세션 로드 오류", f"세션 로드 중 오류 발생: {str(e)}")
    
    def delete_selected_session(self):
        """선택한 세션 삭제"""
        try:
            selection = self.session_listbox.curselection()
            if not selection:
                return
            
            # 확인 메시지
            if not messagebox.askyesno("세션 삭제", "선택한 세션을 삭제하시겠습니까?"):
                return
            
            # 선택한 항목의 세션 ID 가져오기
            index = selection[0]
            session_id = self.session_listbox.itemcget(index, "session_id")
            
            # 세션 파일 경로
            session_file = os.path.join(
                os.path.expanduser("~"), 
                ".ai_terminal", 
                "sessions", 
                f"{session_id}.json"
            )
            
            # 파일 삭제
            if os.path.exists(session_file):
                os.remove(session_file)
            
            # 목록에서 제거
            self.session_listbox.delete(index)
            
            messagebox.showinfo("세션 삭제", "세션이 삭제되었습니다.")
            
        except Exception as e:
            messagebox.showerror("삭제 오류", f"세션 삭제 중 오류 발생: {str(e)}")


# =============================================
# 사용자 정의 도구 예제
# =============================================
def create_example_tool():
    """예제 도구 파일 생성"""
    tools_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal", "tools")
    os.makedirs(tools_dir, exist_ok=True)
    
    example_tool_path = os.path.join(tools_dir, "example_tool.py")
    
    example_code = """
def register_tools(tool_registry):
    
    # 파일 검색 도구
    tool_registry.register_tool(
        "find_files",
        {
            "name": "find_files",
            "description": "특정 패턴의 파일을 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "검색할 파일 패턴 (예: *.txt)"
                    },
                    "directory": {
                        "type": "string",
                        "description": "검색 시작 디렉토리 (기본값: 현재 디렉토리)"
                    }
                },
                "required": ["pattern"]
            }
        },
        find_files_callback
    )

def find_files_callback(args):
    import os
    import glob
    
    pattern = args.get("pattern", "*")
    directory = args.get("directory", os.getcwd())
    
    try:
        # 경로 결합
        search_pattern = os.path.join(directory, pattern)
        
        # 파일 검색
        files = glob.glob(search_pattern)
        
        # 결과 반환
        if files:
            return f"{len(files)}개 파일 찾음: " + ", ".join(files[:10]) + (
                f" 외 {len(files) - 10}개 더..." if len(files) > 10 else "")
        else:
            return f"패턴 '{pattern}'과 일치하는 파일을 찾을 수 없습니다."
    
    except Exception as e:
        return f"파일 검색 중 오류 발생: {str(e)}"
"""
    
    with open(example_tool_path, "w", encoding="utf-8") as f:
        f.write(example_code)
    
    return example_tool_path


# =============================================
# 사용자 정의 LLM 모듈 예제
# =============================================
def create_example_llm_module():
    """예제 LLM 모듈 파일 생성"""
    llm_dir = os.path.join(os.path.expanduser("~"), ".ai_terminal", "llm_modules")
    os.makedirs(llm_dir, exist_ok=True)
    
    example_llm_path = os.path.join(llm_dir, "custom_llm.py")
    
    with open(example_llm_path, "w", encoding="utf-8") as f:
        f.write("""

class LLMService(ABC):
    
    @abstractmethod
    def initialize(self, api_key=None):
        pass
    
    @abstractmethod
    def generate_response(self, messages, tools=None):
        pass
    
    @abstractmethod
    def get_name(self):
        pass


class CustomLLMService(LLMService):
    
    def initialize(self, api_key=None):
        self.api_key = api_key
        
        # 여기에 실제 LLM 클라이언트 초기화 코드 추가
        # 예: self.client = anthropic.Anthropic(api_key=api_key)
        
        # 초기화 성공 여부 반환
        return True
    
    def generate_response(self, messages, tools=None):
        try:
            # 여기에 실제 LLM API 호출 코드 추가
            # 예: response = self.client.messages.create(...)
            
            # 테스트용 더미 응답
            if tools:
                # 함수 호출 예제
                for msg in messages:
                    if msg["role"] == "user" and "명령어" in msg["content"]:
                        return {
                            "content": "명령어를 실행하겠습니다.",
                            "function_call": {
                                "name": "execute_terminal_command",
                                "arguments": '{"command": "dir", "reason": "파일 목록 확인 요청"}'
                            }
                        }
            
            # 일반 응답
            return {"content": "이것은 사용자 정의 LLM 응답입니다."}
            
        except Exception as e:
            return {"content": f"오류 발생: {str(e)}"}
    
    def get_name(self):
        return "사용자 정의 LLM"
""")
    
    return example_llm_path


# =============================================
# 메인 함수
# =============================================
if __name__ == "__main__":
    # 예제 도구 및 LLM 모듈 파일 생성
    create_example_tool()
    create_example_llm_module()
    
    # 터미널 시작
    terminal = AITerminal()
    terminal.mainloop()
# %%

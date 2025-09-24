# app.py
import os
from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings


# ──────────────────────────────────────────────────────────────────────────────
# Env & Embeddings
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # OPENAI_API_KEY 필요 (OpenAIEmbeddings 사용 시)
emb = OpenAIEmbeddings(model="text-embedding-3-small")
PERSIST_DIR = "./db"

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def last_k_turns(history, k=6):
    """Human/AI 메시지 기준 최근 k개만 남겨 토큰 절약."""
    turns = []
    cnt = 0
    for m in reversed(history):
        if isinstance(m, (HumanMessage, AIMessage)):
            turns.append(m)
            cnt += 1
            if cnt >= k:
                break
    return list(reversed(turns))


# ──────────────────────────────────────────────────────────────────────────────
# Prompts (방식 A: 질문 정제 → 검색 → 답변)
# ──────────────────────────────────────────────────────────────────────────────
# 1) 최신 사용자 질문을 '독립 질문'으로 재작성
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite the user's latest question to be self-contained using the chat history as hints. "
     "Return only the rewritten question text."),
    MessagesPlaceholder("chat_history"),
    ("human", "Latest user question: {question}")
])

# 2) 최종 답변: (히스토리 + 검색 컨텍스트) 제공
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use only the given CONTEXT to answer the QUESTION. "
     "If the context is insufficient, say you don't know in Korean."),
    MessagesPlaceholder("chat_history"),
    ("human",
     "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer in Korean, be concise.")
])


# ──────────────────────────────────────────────────────────────────────────────
# Chain builder (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_chain(option_name):
    # 1) 기존 Chroma DB 복구
    if option_name == "청년매입임대":
        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=emb,
            collection_name="YOUTH" 
        )
        gu_li = ['강남구','강동구','강북구','강서구','관악구','광진구','구로구','금천구','노원구','도봉구','동대문구','마포구','서대문구','서초구','성북구','송파구','양천구','영등포구','용산구','은평구','종로구','중구','중랑구']

    elif option_name == "신혼·신생아매입임대Ⅰ":
        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=emb,
            collection_name="MARRY1"
        )
        gu_li = ['강남구','강동구','강북구','강서구','광진구','노원구','도봉구','동대문구','동작구','마포구','서대문구','서초구','성북구','송파구','양천구','영등포구','은평구','중랑구']

    elif option_name ==  "신혼·신생아_매입임대주택Ⅱ":
        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=emb,
            collection_name="MARRY2"
        )
        gu_li = ['강동구','강북구','강서구','광진구','구로구','금천구','노원구','도봉구','동대문구','동작구','서대문구','서초구','성북구','송파구','양천구','영등포구','은평구','중랑구']
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 2) LLM (Ollama)
    llm = ChatOllama(model="eeve-q5km:latest", temperature=0.2, streaming=True)

    # 3) 독립 질문 생성기 (입력: {"question": str, "chat_history": [BaseMessage,...]} → 출력: str)
    make_standalone = CONDENSE_PROMPT | llm | StrOutputParser()

    # 4) 파이프라인
    #    단계A: standalone_question 생성 + chat_history 보존  → 키 확정 (RunnableParallel로 안전하게)
    stepA = RunnableParallel(
        standalone_question={
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        } | make_standalone,
        chat_history=itemgetter("chat_history"),
    )

    #    단계B: 검색 컨텍스트/질문/히스토리 묶기 → ANSWER_PROMPT
    stepB = RunnableParallel(
        context=itemgetter("standalone_question") | retriever | format_docs,
        question=itemgetter("standalone_question"),
        chat_history=itemgetter("chat_history"),
    )

    # 5) 전체 RAG 체인 (스트리밍 시 문자열 chunk를 반환)
    rag_chain = stepA | stepB | ANSWER_PROMPT | llm | StrOutputParser()
    return rag_chain, gu_li, vector_store

def set_filter(vector_store, filter):
    retriever = vector_store.as_retriever(
    search_kwargs={"k": 10, "filter": filter}
    )
    # 2) LLM (Ollama)
    llm = ChatOllama(model="eeve-q5km:latest", temperature=0.2, streaming=True)

    # 3) 독립 질문 생성기 (입력: {"question": str, "chat_history": [BaseMessage,...]} → 출력: str)
    make_standalone = CONDENSE_PROMPT | llm | StrOutputParser()

    # 4) 파이프라인
    #    단계A: standalone_question 생성 + chat_history 보존  → 키 확정 (RunnableParallel로 안전하게)
    stepA = RunnableParallel(
        standalone_question={
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        } | make_standalone,
        chat_history=itemgetter("chat_history"),
    )

    #    단계B: 검색 컨텍스트/질문/히스토리 묶기 → ANSWER_PROMPT
    stepB = RunnableParallel(
        context=itemgetter("standalone_question") | retriever | format_docs,
        question=itemgetter("standalone_question"),
        chat_history=itemgetter("chat_history"),
    )

    # 5) 전체 RAG 체인 (스트리밍 시 문자열 chunk를 반환)
    rag_chain = stepA | stepB | ANSWER_PROMPT | llm | StrOutputParser()

    return rag_chain

def reset_ragchain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 2) LLM (Ollama)
    llm = ChatOllama(model="eeve-q5km:latest", temperature=0.2, streaming=True)

    # 3) 독립 질문 생성기 (입력: {"question": str, "chat_history": [BaseMessage,...]} → 출력: str)
    make_standalone = CONDENSE_PROMPT | llm | StrOutputParser()

    # 4) 파이프라인
    #    단계A: standalone_question 생성 + chat_history 보존  → 키 확정 (RunnableParallel로 안전하게)
    stepA = RunnableParallel(
        standalone_question={
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        } | make_standalone,
        chat_history=itemgetter("chat_history"),
    )

    #    단계B: 검색 컨텍스트/질문/히스토리 묶기 → ANSWER_PROMPT
    stepB = RunnableParallel(
        context=itemgetter("standalone_question") | retriever | format_docs,
        question=itemgetter("standalone_question"),
        chat_history=itemgetter("chat_history"),
    )

    # 5) 전체 RAG 체인 (스트리밍 시 문자열 chunk를 반환)
    rag_chain = stepA | stepB | ANSWER_PROMPT | llm | StrOutputParser()

    return rag_chain
    

def reset_session_and_cache():
    # 채팅 이력 초기화
    st.session_state["lc_messages"] = [AIMessage(content="무엇을 도와드릴까요?")]
    # 체인 캐시 무효화 후 즉시 리런
    build_chain.clear()
    st.rerun()

with st.sidebar:
    st.header("⚙️ 유형 설정")
    option_name = st.selectbox(
        "유형",
        options=["청년매입임대", "신혼·신생아매입임대Ⅰ", "신혼·신생아_매입임대주택Ⅱ"],
        index=0,
        key="option_name",
        on_change=reset_session_and_cache
    )

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.header("💬Chat bot")
rag_chain, gu_li, vector_store = build_chain(option_name)

st.write(vector_store._collection_name)
# 세션 상태 초기화 (UI엔 간단 인사만 표시)
if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = [AIMessage(content="무엇을 도와드릴까요?")]

# 대화 렌더링 (SystemMessage는 화면에서 생략)
for m in st.session_state.lc_messages:
    if isinstance(m, SystemMessage):
        continue
    role = "assistant" if isinstance(m, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(m.content)

# 입력
user_input = st.chat_input("질문을 입력해주세요 :")
if user_input:
    for gu in gu_li:
        if gu in user_input:
            filters = {"구": gu} 
            rag_chain = set_filter(vector_store, filters)
        else:
            rag_chain = reset_ragchain(vector_store)
           
    # 1) 사용자 메시지 이력에 추가 & 렌더
    st.session_state.lc_messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) 히스토리 준비(시스템 메시지 제외, 최근 k턴만 유지)
    chat_history = [m for m in st.session_state.lc_messages if not isinstance(m, SystemMessage)]
    chat_history = last_k_turns(chat_history, k=6)

    # 3) 스트리밍 응답
    with st.chat_message("assistant"):
        box = st.empty()
        buf = ""
        with st.spinner("Chat bot이 답변을 생성하는 중 ..."):
            # 입력은 항상 {"question": <str>, "chat_history": <list>} 형태
            for chunk in rag_chain.stream({
                "question": user_input,
                "chat_history": chat_history,
                "k": 10
            }):
                buf += chunk              # StrOutputParser 사용 → chunk는 문자열
                box.markdown(buf)

    # 4) 최종 답변 이력에 추가
    st.session_state.lc_messages.append(AIMessage(content=buf))


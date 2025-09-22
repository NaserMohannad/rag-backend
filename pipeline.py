import os
import weaviate
import requests
from weaviate.classes.init import Auth
from weaviate.classes.query import Rerank
from transformers import AutoTokenizer, pipeline
import torch

# 1. الاتصال بقاعدة Weaviate
def connect_to_db():
    headers = {
        "X-Cohere-Api-Key": os.getenv("COHERE_API_KEY")
    }

    weaviate_url = os.getenv("veaviat_rest")
    weaviate_api_key = os.getenv("weaviat_api_key")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        headers=headers
    )

    return client, client.is_ready()


# 2. إعداد الـ Router model
model_checkpoint = "EN3IMI/RouterAraBERT"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
classifier = pipeline("sentiment-analysis", model=model_checkpoint)


# 3. البحث في FAQ
def search_for_faq(user_query, client):
    collection = client.collections.use("FAQ")
    response = collection.query.hybrid(
        query=user_query,
        limit=10,
        alpha=0.40,
        rerank=Rerank(
            prop="content",
            query=user_query,
        )
    )

    queries = []
    for idx, obj in enumerate(response.objects[:2]):
        message_to_router = f"{user_query} [SEP] {obj.properties['content']}"
        queries.append(message_to_router)

    return queries, response.objects[:3]


# 4. البحث في Laws
def search_for_laws(user_query, client):
    collection = client.collections.use("Laws")
    response = collection.query.hybrid(
        query=user_query,
        limit=10,
        alpha=0.40,
        rerank=Rerank(
            prop="text",
            query=user_query
        )
    )
    return response.objects[:3]


# 5. اتخاذ القرار بالـ Router
def router_decision(queries, classifier):
    results = classifier(queries)
    labels = [res['label'] for res in results]
    numeric_labels = [1 if label == "LABEL_1" else 0 for label in labels]
    return any(numeric_labels)


# 6. LLM Response للـ FAQ
def llm_response_faq(query, docs):
    chunks = [i.properties["content"] for i in docs]

    API_KEY = os.getenv("perplixty_api")
    ENDPOINT = "https://api.perplexity.ai/chat/completions"

    system_prompt = """
    You are an intelligent assistant specialized in the Jordanian Land and Survey Department.
    Your task is to provide answers strictly based on the context provided from FAQ files.
    Guidelines:
    1. Use only the information available in the provided files. Do not hallucinate or invent any information.
    2. If the provided context does not contain a relevant answer to the user's question, respond with: "I do not know the answer."
    3. Correct any spelling or typographical errors present in the extracted text from the files.
    4. Provide answers in Arabic when possible.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Context:\n" + "\n".join(chunks) + "\n\nQuestion:\n" + query}
    ]

    data = {"model": "sonar-pro", "messages": messages, "max_tokens": 250, "temperature": 0.5}
    resp = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json=data)
    responsed = resp.json()
    return responsed["choices"][0]["message"]["content"]


# 7. LLM Response للـ Laws
def llm_response_laws(query, docs):
    chunks = [i.properties["text"] for i in docs]

    API_KEY = os.getenv("perplixty_api")
    ENDPOINT = "https://api.perplexity.ai/chat/completions"

    system_prompt = """
    You are an intelligent assistant specialized in Jordanian laws and legislation.
    Your task is to provide answers strictly based on the context provided from the legal documents.
    Guidelines:
    1. Use only the information available in the provided files. Do not hallucinate.
    2. If the provided context does not contain a relevant answer, respond in Arabic with: "لا أعلم الجواب".
    3. Provide answers in Arabic when possible.
    4. Be accurate and concise.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Context:\n" + "\n".join(chunks) + "\n\nQuestion:\n" + query}
    ]

    data = {"model": "sonar-pro", "messages": messages, "max_tokens": 300, "temperature": 0.5}
    resp = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json=data)
    responsed = resp.json()
    return responsed["choices"][0]["message"]["content"]


# 8. النظام الذكي المدمج
def IntelligentRAGSystem(query, client, classifier):
    queries_bert, docs_faq_llm = search_for_faq(query, client)
    decision = router_decision(queries_bert, classifier)
    if decision:
        return llm_response_faq(query, docs_faq_llm)
    else:
        docs_law_llm = search_for_laws(query, client)
        return llm_response_laws(query, docs_law_llm)


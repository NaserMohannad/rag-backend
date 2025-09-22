import weaviate
import requests
import json
from google.colab import userdata
from weaviate.classes.init import Auth
from weaviate.classes.query import Rerank
from transformers import (
    AutoTokenizer,
    pipeline
)

def connect_to_db():
    headers = {
        "X-Cohere-Api-Key": userdata.get('COHERE_API_KEY')
    }

    weaviate_url = userdata.get('veaviat_rest')
    weaviate_api_key = userdata.get('weaviat_api_key')

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        headers=headers
    )

    return client, client.is_ready()

model_checkpoint = "EN3IMI/RouterAraBERT"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

classifier = pipeline("sentiment-analysis", model=model_checkpoint)


def search_for_faq(user_query, client):
    collection = client.collections.use("FAQ")

    response = collection.query.hybrid(
        query=user_query,
        limit=10,
        alpha = 0.40,
        rerank=Rerank(
            prop="content",
            query=user_query,
        )
    )

    queries = []

    for idx, obj in enumerate(response.objects[:2]):
        message_to_router = f"{user_query} [SEP] {obj.properties["content"]}"
        queries.append(message_to_router)

    return queries, response.objects[:3]

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

def router_decision(queries, classifier):
  results = classifier(queries)
  labels = [res['label'] for res in results]
  numeric_labels = [1 if label == 'LABEL_1' else 0 for label in labels]

  return any(numeric_labels)


def llm_response_faq(query, docs):
  chunks = []
  for i in docs:
    chunks.append(i.properties["content"])

  API_KEY = userdata.get('perplixty_api')
  ENDPOINT = "https://api.perplexity.ai/chat/completions"

  system_prompt = """
  You are an intelligent assistant specialized in the Jordanian Land and Survey Department. Your task is to provide answers strictly based on the context provided from FAQ files.

  Guidelines:

  1. Use only the information available in the provided files. Do not hallucinate or invent any information.
  2. If the provided context does not contain a relevant answer to the user's question, respond with: "I do not know the answer."
  3. Correct any spelling or typographical errors present in the extracted text from the files.
  4. Provide brief clarifications or explanations only when necessary to make the answer clear, but do not add new facts.
  5. Do not modify the facts or data from the files; respect the sensitivity of the information.
  6. Focus only on questions related to Jordanian land, survey, and administrative data.
  7. Answer in the language of the user's question. Most questions will be in Arabic, so prioritize answering in Arabic when possible.

  Instructions for answering:

  - First, identify the most relevant FAQ entry based on the user's question.
  - Then, provide the answer exactly as it appears in the file, fixing only spelling mistakes and minor formatting issues.
  - AVOID PROVIDIND INORMATION NOT PRESENT IN THE CONTEXT.
  - Always maintain accuracy and reliability.
  - If you don't know the answer tell the user that you don't know in Arabic
  """


  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Context:\n" + "\n".join(chunks) + "\n\nQuestion:\n" + query}
  ]

  data = {
    "model": "sonar-pro",
    "messages": messages,
    "max_tokens": 250,
    "temperature": 0.5
  }

  resp = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json=data)
  responsed = resp.json()
  return responsed['choices'][0]['message']['content']


def llm_response_laws(query, docs):
    chunks = []
    for i in docs:
        chunks.append(i.properties["text"])

    API_KEY = userdata.get('perplixty_api')
    ENDPOINT = "https://api.perplexity.ai/chat/completions"

    system_prompt = """
    You are an intelligent assistant specialized in Jordanian laws and legislation.
    Your task is to provide answers strictly based on the context provided from the legal documents.

    Guidelines:
    1. Use only the information available in the provided files. Do not hallucinate.
    2. If the provided context does not contain a relevant answer, respond in Arabic with: "لا أعلم الجواب".
    3. Correct minor spelling/formatting mistakes if needed.
    4. Provide answers in Arabic when possible.
    5. Be accurate and concise.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Context:\n" + "\n".join(chunks) + "\n\nQuestion:\n" + query}
    ]

    data = {
        "model": "sonar-pro",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.5
    }

    resp = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=data
    )
    responsed = resp.json()
    return responsed['choices'][0]['message']['content']


def IntelligentRAGSystem(query, client, classifier):
  queries_bert, docs_faq_llm = search_for_faq(query, client)
  decision = router_decision(queries_bert, classifier)
  if decision == True:
    llm_response = llm_response_faq(query, docs_faq_llm)
  else:
    docs_law_llm = search_for_laws(query, client)
    llm_response = llm_response_laws(query, docs_law_llm)

  return llm_response
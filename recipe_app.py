import streamlit as st #Uygulama için 
from haystack import Pipeline #Pipeline oluşturmak için oluşturulan Pipeline ile RAG akışı arasında bağlantı kurulur.
from haystack.document_stores.in_memory import InMemoryDocumentStore #Dökümanların depolanması için ek bir servise ya da bağımlılığa gerek kalmadan depolamayı sağlar. Production için tavsiye edilmez.
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever #Embeddinglerden alakalı olanları çekmek için
from haystack.components.converters import PyPDFToDocument #PDF dosyasını dökümana dönüştürmek için
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder #Veriyi embed etmek için
from haystack.dataclasses import ChatMessage #Chat Message yapısı oluşturmak için
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder #Chatpromptu oluşturmak için
#Gemini için
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator
#Agentic search için
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.components.routers import ConditionalRouter
from dotenv import load_dotenv #API key i yüklemek için 

#Başlık oluşturalım.
st.title('Gemini Tabanlı RAG ile Türk Mutfağı Tarif Uygulaması')

#Önce dökumanı oluşturalım ve API keyleri çekelim.
document_store = InMemoryDocumentStore()
load_dotenv()
pdf_path = 'turk-mutfagi-kitap.pdf'

#Embedder
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

#Elimizdeki pdf dosyasını dökümana çevirip embedderını döküman üzerine uygulayalım.
pdf_converter = PyPDFToDocument()
documents = pdf_converter.run(sources=[pdf_path])["documents"]
docs_with_embeddings = doc_embedder.run(documents=documents)
document_store.write_documents(docs_with_embeddings["documents"])

#Retriever oluşturalım ve gelen sorguyu embed etmesi için bir TextEmbedder oluşturalım.
text_embedder = SentenceTransformersTextEmbedder('sentence-transformers/all-MiniLM-L6-v2')
retriever = InMemoryEmbeddingRetriever(document_store)

prompt_template = [
    ChatMessage.from_user(
        """
Answer the following query given the documents.
If the answer is not contained within the documents reply with 'cevap_yok'
Answer all questions in Turkish.

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}
Query: {{query}}
"""
    )
]

prompt_builder = ChatPromptBuilder(template=prompt_template, required_variables="*")
llm = GoogleAIGeminiChatGenerator(model="gemini-2.0-flash")

prompt_for_websearch = [
    ChatMessage.from_user(
        """
Answer the following query given the documents retrieved from the web.
Your answer shoud indicate that your answer was generated from websearch.
Answer all questions in Turkish.

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}

Query: {{query}}
"""
    )
]

websearch = SerperDevWebSearch()
prompt_builder_for_websearch = ChatPromptBuilder(template=prompt_for_websearch, required_variables="*")
llm_for_websearch = GoogleAIGeminiChatGenerator(model="gemini-2.0-flash")

#İnternet yönlendirmesi oluşturalım.
main_routes = [
    {
        "condition": "{{'cevap_yok' in replies[0].text}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch",
        "output_type": str,
    },
    {
        "condition": "{{'cevap_yok' not in replies[0].text}}",
        "output": "{{replies[0].text}}",
        "output_name": "answer",
        "output_type": str,
    },
]
router = ConditionalRouter(main_routes)


# Session state: mesaj geçmişi\ if "messages" not in st.session_state:
st.session_state.messages = [
    {"role": "system", "content": ""}
]


#Kullanıcı mesajı girişi
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Mesajınızı yazın:", "")
    submit = st.form_submit_button("Gönder")

if submit and user_input:
    # Kullanıcı mesajını kaydet
    st.session_state.messages.append({"role": "user", "content": user_input})

    agentic_rag_pipe = Pipeline()
    agentic_rag_pipe.add_component("embedder", text_embedder)
    agentic_rag_pipe.add_component("retriever", retriever)
    agentic_rag_pipe.add_component("prompt_builder", prompt_builder)
    agentic_rag_pipe.add_component("llm", llm)
    agentic_rag_pipe.add_component("router", router)
    agentic_rag_pipe.add_component("websearch", websearch)
    agentic_rag_pipe.add_component("prompt_builder_for_websearch", prompt_builder_for_websearch)
    agentic_rag_pipe.add_component("llm_for_websearch", llm_for_websearch)
    
    agentic_rag_pipe.connect("embedder", "retriever")
    agentic_rag_pipe.connect("retriever", "prompt_builder.documents")
    agentic_rag_pipe.connect("prompt_builder.prompt", "llm.messages")
    agentic_rag_pipe.connect("llm.replies", "router.replies")
    agentic_rag_pipe.connect("router.go_to_websearch", "websearch.query")
    agentic_rag_pipe.connect("router.go_to_websearch", "prompt_builder_for_websearch.query")
    agentic_rag_pipe.connect("websearch.documents", "prompt_builder_for_websearch.documents")
    agentic_rag_pipe.connect("prompt_builder_for_websearch", "llm_for_websearch")
    
    query = user_input

    result = agentic_rag_pipe.run(
        {"embedder": {"text": query}, "prompt_builder": {"query": query}, "router": {"query": query}}
    )
    
    if result.get('websearch') is None:
        assistant_message = result['router']['answer']
    else:
        reply_msg = result['llm_for_websearch']['replies'][0]
        assistant_message = reply_msg._content[0].text, result['websearch']['links']

    # Asistan mesajını kaydet
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

# Ana alanda son mesaj göster
if st.session_state.messages:
    last = st.session_state.messages[-1]
    #prefix = "Siz" if last["role"] == "user" else "Asistan"
    #st.markdown(f"**{prefix}:** {last['content']}")
    st.markdown(f" {last['content']}")
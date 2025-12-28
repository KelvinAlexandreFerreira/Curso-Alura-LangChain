import os
import time
from gemini_setup import get_gemini_llm
from langchain.globals import set_debug
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from prompts_constants import PROMPT_RAG_PT 

# --- 1. Configuração Inicial ---
llm = get_gemini_llm()
set_debug(True)

# --- 2. Carregamento e Divisão do Texto ---
nome_arquivo = "GTB_gold_Dez_25.txt"

# Obtém o caminho absoluto para onde o script está olhando
caminho_completo = os.path.abspath(nome_arquivo)
print(f"📂 Procurando arquivo em: {caminho_completo}")

# Verificação: Se não achar, para tudo.
if not os.path.exists(nome_arquivo):
    print("\n❌ ERRO CRÍTICO: O arquivo não foi encontrado!")
    exit()

carregador = TextLoader(nome_arquivo, encoding="utf8")
documentos = carregador.load()

# O RecursiveSplitter garante que o texto será quebrado corretamente
quebrador = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""] # Força bruta se necessário
)
textos = quebrador.split_documents(documentos)

print(f"\n✅ Sucesso! Foram gerados {len(textos)} fragmentos de texto.")

# --- 3. Geração de Embeddings e Vector Store (COM RETRY AUTOMÁTICO) ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print("\n--- Gerando Embeddings em Lotes com Retry Inteligente ---")
print(f"--- Modelo Utilizado: {embeddings.model} ---")

vectorstore = None
tamanho_lote = 10  # Processa 10 fragmentos por vez

# Loop para processar os fragmentos aos poucos
for i in range(0, len(textos), tamanho_lote):
    lote_atual = textos[i:i + tamanho_lote]
    print(f"🔹 Processando lote {i // tamanho_lote + 1} (Fragmentos {i} a {i + len(lote_atual)})...")
    
    # Loop Infinito de Tentativa (Retry) para este lote específico
    while True:
        try:
            if vectorstore is None:
                # Primeiro lote cria o banco
                vectorstore = FAISS.from_documents(lote_atual, embeddings)
            else:
                # Lotes seguintes são adicionados ao banco existente
                vectorstore.add_documents(lote_atual)
            
            # Se deu certo, sai do while e vai para o próximo lote do for
            break 
            
        except Exception as e:
            # Verifica se é erro de cota (429)
            if "429" in str(e) or "ResourceExhausted" in str(e):
                print(f"⚠️ Cota do Google excedida! Esperando 60 segundos para tentar novamente...")
                time.sleep(60) # Espera 1 minuto inteiro
            else:
                # Se for outro erro (ex: internet caiu), mostra o erro e para
                raise e
    
    # Pequena pausa extra entre lotes de sucesso
    if i + tamanho_lote < len(textos):
        time.sleep(2)

print("\n✅ Todos os embeddings foram gerados com sucesso!")

# --- 4. Criação da Chain de RAG ---
retriever = vectorstore.as_retriever()
document_chain = create_stuff_documents_chain(llm, PROMPT_RAG_PT)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 5. Execução ---
pergunta = "Como devo proceder caso tenho um item comprado roubado?"

print(f"\n--- Perguntando: {pergunta} ---")
resposta = retrieval_chain.invoke({"input": pergunta})

print("\n--- Resposta ---")
print(resposta["answer"])
import os
from dotenv import load_dotenv

# Importação da integração do Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Carrega variáveis de ambiente
load_dotenv()

# --- CONFIGURAÇÃO AVANÇADA DO MODELO ---
# Configuração para o Gemini 3 Flash Preview.
configuracao_geracao = {
    "temperature": 0.7,        
    "top_p": 0.95,            
    "top_k": 40,              
    "max_output_tokens": 8192,
}

llm = ChatGoogleGenerativeAI(
    # Atualizado para o modelo que apareceu na sua lista
    model="gemini-3-flash-preview", 
    google_api_key=os.getenv("GEMINI_API_KEY"),
    
    # --- OTIMIZAÇÃO DE REQUISIÇÕES ---
    # max_retries=0 significa: "Tente uma vez. Se der erro, pare imediatamente."
    # Isso impede que o LangChain faça várias chamadas em background e estoure seu limite.
    max_retries=0,            
    
    # Passando as configurações
    temperature=configuracao_geracao["temperature"],
    top_p=configuracao_geracao["top_p"],
    top_k=configuracao_geracao["top_k"],
    max_output_tokens=configuracao_geracao["max_output_tokens"]
)

# --- DADOS DE ENTRADA ---
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

# --- DEFINIÇÃO DO PROMPT ---
template = (
    "Crie um roteiro de viagem de {dias} dias, "
    "para uma família com {criancas} crianças, "
    "que gostam de {atividade}."
)

prompt = PromptTemplate.from_template(template)

# --- A CHAIN ---
chain = prompt | llm

# --- EXECUÇÃO ---
print(f"Gerando roteiro com {llm.model}...")

try:
    resposta = chain.invoke({
        "dias": numero_de_dias,
        "criancas": numero_de_criancas,
        "atividade": atividade
    })

    print("\n--- Resposta do Gemini ---")
    print(resposta.content)

except Exception as e:
    print(f"\n❌ Erro: {e}")
    # Dica de debug caso o modelo 3 exija algo específico no futuro
    print("Dica: Se receber erro de 'not found', confirme se 'gemini-3-flash-preview' ainda consta no script verificar_modelos.py")
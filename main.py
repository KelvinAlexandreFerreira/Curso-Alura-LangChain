import os
from dotenv import load_dotenv

# Importação da integração do Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Carrega variáveis de ambiente
load_dotenv()

# --- CONFIGURAÇÃO DO MODELO (O "MOTOR") ---
# Aqui é onde você trocaria se quisesse voltar para OpenAI.
# Em vez de ChatOpenAI, usamos ChatGoogleGenerativeAI.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"), # Certifique-se de ter essa chave no .env
    temperature=0.7 # Criatividade (0 a 1)
)

# --- DADOS DE ENTRADA ---
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

# --- DEFINIÇÃO DO PROMPT (A "CARROCERIA") ---
# Isso aqui não muda, independente se o motor é Google, OpenAI ou Meta.
template = (
    "Crie um roteiro de viagem de {dias} dias, "
    "para uma família com {criancas} crianças, "
    "que gostam de {atividade}."
)

prompt = PromptTemplate.from_template(template)

# --- A CHAIN (LIGANDO TUDO) ---
# Usamos a sintaxe moderna (LCEL) com o pipe "|"
chain = prompt | llm

# --- EXECUÇÃO ---
print(f"Gerando roteiro para {atividade}...")
resposta = chain.invoke({
    "dias": numero_de_dias,
    "criancas": numero_de_criancas,
    "atividade": atividade
})

print("\n--- Resposta do Gemini ---")
print(resposta.content)
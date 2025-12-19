from langchain_core.prompts import PromptTemplate

# Importamos a função do nosso novo módulo separado
from gemini_setup import get_gemini_llm

# --- INICIALIZAÇÃO DO MODELO ---
# Agora basta chamar essa função para ter o LLM pronto!
llm = get_gemini_llm()

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
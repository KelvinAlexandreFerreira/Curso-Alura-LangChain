import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Garante que as variáveis de ambiente sejam carregadas ao importar este módulo
load_dotenv()

def get_gemini_llm():
    """
    Cria e retorna uma instância configurada do ChatGoogleGenerativeAI.
    Use esta função em qualquer arquivo .py para obter o modelo pronto.
    """
    
    # --- CONFIGURAÇÃO AVANÇADA ---
    configuracao_geracao = {
        "temperature": 0.7,        
        "top_p": 0.95,            
        "top_k": 40,              
        "max_output_tokens": 8192,
    }

    # Inicializa o modelo
    llm = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it", 
        google_api_key=os.getenv("GEMINI_API_KEY"),
        
        # Otimização de requisições para evitar bloqueio
        max_retries=0,            
        
        # Aplica as configurações
        temperature=configuracao_geracao["temperature"],
        top_p=configuracao_geracao["top_p"],
        top_k=configuracao_geracao["top_k"],
        max_output_tokens=configuracao_geracao["max_output_tokens"]
    )
    
    return llm
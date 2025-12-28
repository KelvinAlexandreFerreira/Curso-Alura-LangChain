from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# --- Template para Conversa Geral (ConversationChain) ---
_TEMPLATE_CONVERSA = """A seguir está uma conversa amigável entre um humano e uma IA. 
A IA é falante e fornece muitos detalhes específicos de seu contexto. 
Se a IA não souber a resposta para uma pergunta, ela diz sinceramente que não sabe.
A IA deve responder sempre em Português do Brasil.

Conversa atual:
{history}
Humano: {input}
IA:"""

PROMPT_CONVERSA_PT = PromptTemplate(
    input_variables=["history", "input"], 
    template=_TEMPLATE_CONVERSA
)

# --- Template para Resumo de Memória (ConversationSummaryMemory) ---
_TEMPLATE_SUMARIO = """Resuma progressivamente as linhas de conversa abaixo, adicionando ao resumo anterior e retornando um novo resumo.

Resumo atual:
{summary}

Nova linha de conversa:
{new_lines}

Novo resumo (em Português do Brasil):"""

PROMPT_SUMARIO_PT = PromptTemplate(
    input_variables=["summary", "new_lines"], 
    template=_TEMPLATE_SUMARIO
)

# --- Template para RAG (Retrieval-Augmented Generation) ---
_TEMPLATE_RAG = """Você é um assistente útil.
Responda à pergunta com base APENAS no seguinte contexto fornecido:

{context}

Pergunta: {input}
"""

# Usamos ChatPromptTemplate aqui pois o RAG geralmente usa modelos de chat
PROMPT_RAG_PT = ChatPromptTemplate.from_template(_TEMPLATE_RAG)
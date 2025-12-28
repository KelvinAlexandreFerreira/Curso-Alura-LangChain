from langchain.prompts import PromptTemplate

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
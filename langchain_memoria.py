from langchain_core.prompts import PromptTemplate
from gemini_setup import get_gemini_llm
from langchain.globals import set_debug
from langchain_core.output_parsers import StrOutputParser

llm = get_gemini_llm()
set_debug(True)

mensagens = [
        "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
        "Qual é o melhor período do ano para visitar em termos de clima?",
        "Quais tipos de atividades ao ar livre estão disponíveis?",
        "Alguma sugestão de acomodação eco-friendly por lá?",
        "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
        "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

longa_conversa = ""

for mensagem in mensagens:
    longa_conversa += f"Usuário: {mensagem}\n"
    longa_conversa += f"AI: "

    modelo = PromptTemplate(template=longa_conversa, input_variables=[])
    cadeia = modelo | llm | StrOutputParser()
    resposta = cadeia.invoke(input={})
    #print(resposta)

    longa_conversa += f"{resposta}\n"
    print(longa_conversa)
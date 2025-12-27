from langchain.chains import ConversationChain
from gemini_setup import get_gemini_llm
from langchain.globals import set_debug
from langchain.memory import ConversationBufferWindowMemory

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

memory = ConversationBufferWindowMemory(k=2)
conversation = ConversationChain(llm=llm, verbose=True, memory=memory)

for mensagem in mensagens:
    resposta = conversation.predict(input=mensagem)

    print(resposta)

print(memory.load_memory_variables({}))
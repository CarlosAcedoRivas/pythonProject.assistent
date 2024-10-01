import streamlit as st
from chatbot import predict_class, get_response, intents
from PIL import Image
img= Image.open("logoilla.png")
st.set_page_config(page_title='ILLAGPT', page_icon=img)
def main():
    st.title("📚Assistent virtual illa de rodes📚")
    st.warning("De moment, només es pot fer consultes en català, disculpeu les molèsties,s alumnes de 2BATX🙏🙏! És important que tinguis en compte que l'assistent no et farà els deures (no resol operacions ni crea texts), el que pot fer és proporcionar-te apunts que et podrien ser útils. Pots demanar el que necessitis a continuació!")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "first_message" not in st.session_state:
        st.session_state.first_message = True

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.first_message:
        with st.chat_message("assistant"):
            st.markdown("Hola, com et puc ajudar?")

        st.session_state.messages.append({"role": "assistant", "content": "Hola, com et puc ajudar?"})
        st.session_state.first_message = False

    if prompt := st.chat_input("Escriu aquí un dubte."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        #IMPLEMENTACIÓ DE LA INTEL·LIGÈNCIA ARTIFICIAL
        insts = predict_class(prompt)
        res = get_response(insts, intents)

        with st.chat_message("assistant"):
            st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})

if __name__ == '__main__':
    main()

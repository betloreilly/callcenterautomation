
from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import streamlit as st
from audiorecorder import audiorecorder
import streamlit as st
from PIL import Image
import openai
from tools import AudioSearch
import os

from dotenv import dotenv_values

##############################
### initialize agent #########
##############################

tools = [AudioSearch()]
config = dotenv_values('.env')
openai_key = config['OPENAI_API_KEY']
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key=openai_key,
    temperature=0,
    model_name="gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

# set title
st.title('Call Center Support')
image = Image.open('/Users/betuloreilly/demos/agentswithaudio/source/logo.png')
#st.image(image)
audio = audiorecorder("Click to record", "Recording...")
    

if len(audio) > 0:
    # To play audio in frontend:
    st.audio(audio.tobytes())
     # To save audio to a file:
    wav_file = open("test.mp3", "wb")
    wav_file.write(audio.tobytes())

    audio_file= open("/Users/betuloreilly/llmdemos/agentswithaudio/test.mp3", "rb")
    user_question = openai.Audio.transcribe("whisper-1", audio_file)


    ##############################
    ### compute agent response ###
    ##############################
    if user_question and user_question != "":
        with st.spinner(text="In progress..."):
            response = agent.run('{}, {}'.format(user_question, user_question))
            #st.write(user_question)
            st.write(response)

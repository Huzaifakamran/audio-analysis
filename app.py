import streamlit as st
import speech_recognition as sr
from pydub import utils,AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def save_uploaded_audio(uploaded_file, save_path):
#     audio_content = uploaded_file.getvalue()
#     with open(save_path, "wb") as f:
#         f.write(audio_content)

# def get_prober_name():
#     return "ffmpeg/bin/ffprobe.exe"

def delete_all_files_in_directory(directory_path):
    try:
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # Check if it is a file (not a directory) and delete it
            if os.path.isfile(file_path):
                os.remove(file_path)

        print(f"All files in '{directory_path}' deleted successfully.")

    except Exception as e:
        st.write(f"An error occurred: {e}")

def convert_audio_to_text(input_path,output_dir,max_size_mb=25):
    with st.spinner('converting audio to the standard format'):
        print('inside fun')
        print(input_path.size)
        # AudioSegment.converter = "ffmpeg/bin/ffmpeg.exe"                  
        # utils.get_prober_name = get_prober_name
        audio = AudioSegment.from_file(input_path)
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path.name))[0] + ".wav")
        audio.export(output_path, format="wav")
    
    with st.spinner('splitting audio'):
        # Check file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        print(file_size_mb)
        if file_size_mb <= max_size_mb:
            print('under if')
            parts = [output_path]  # Return a list with the path of the converted .wav file

        else:
            print('under else')
            parts = []
            segment_size_ms = 100000  # Adjust this value based on your needs (10 seconds in this example)
            print(len(audio))
            for start in range(0, len(audio), segment_size_ms):
                end = start + segment_size_ms
                part = audio[start:end]
                part_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path.name))[0]}_{start}-{end}.wav")
                part.export(part_path, format="wav")
                parts.append(part_path)

    print('PARTS: ',len(parts))
    print('PARTS: ',parts)
    with st.spinner(f'extracting text from {len(parts)} audio files'):
        text = ""
        for i in parts:
            print(i)
            audio_file = open(i, "rb")
            transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
            )
            text += transcript.text + " "
        
        return text

def main():
    try:
        st.title("Audio to Text Conversion App")

        if "audio_file" not in st.session_state:
            st.session_state.audio_file = None

        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

        if uploaded_file is not None:
            st.session_state.audio_file = uploaded_file
            st.audio(st.session_state.audio_file, format="audio/wav", start_time=0)

            if st.button("Convert to Text"):
                TranscriptText = convert_audio_to_text(st.session_state.audio_file,'audios')
                # converted_files  = convert_audio_to_text(st.session_state.audio_file,'audios')
                # print("Converted and/or split files:", converted_files)
                # Add tabs
                text,duration = st.tabs(["Audio To Text","Duration"])
        
                with text:
                    # for utterance in transcript.utterances:
                    #     st.write(f"Speaker {utterance.speaker}: {utterance.text}")
                    st.write(TranscriptText)
                
                with duration:
                    st.write("In Progress...")

                delete_all_files_in_directory('audios')
                # with concepts:
                #     assistant = client.completions.create(
                #     model="gpt-3.5-turbo-instruct",
                #     prompt=f'''You need to give 3 details by checking the below text extracted 
                #     from an audio in spanish and in each audio mostly there will be 2 people one is client
                #     and the other one is salesman. 
                #     The first detail is "Concept", you need to understand the concept spoken in the audio.
                #     For example customer talk about buying a car and talk about speed, 
                #     engine, seat, security, etc so you need to understand the converation and 
                #     which part is said by who (client & salesman). 
                #     The second detail is "Client Profile", you need to identify the client profile based on
                #     the following preferences of purchasing: [Trend,Money,Confort, Fidelity,Security and Pride]. 
                #     And the last detail is to called "Probablity", you need to analyze the conversation based on client profile 
                #     and establish a probability of purchasing from 0 to 1.

                #     In the output each detail should be in the new line like this:
                #     Concept:
                #     Client Profile:
                #     Probablity:

                #     Please find the audio text below:\n
                #     {transcript.text}''',
                #     max_tokens=1500,
                #     temperature=0
                #     )
                #     result = assistant.choices[0].text
                #     print(result)
                #     st.write(result)
    except Exception as e:
        st.write(e)         

                
if __name__ == "__main__":
    main()

import streamlit as st
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess
import math
import spacy
import re
from collections import Counter
import pandas as pd
load_dotenv()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_prober_name():
#     return "ffmpeg/bin/ffprobe.exe"

def extract_nouns_with_counts(glocary,black_list,TranscriptText,brand_list):
    nlp = spacy.load('es_core_news_sm')
    def extract_nouns(text,original_text):
        doc = nlp(text)
        nouns = []
        # nouns = [token.text for token in doc if token.pos_ == 'NOUN' and token.text not in black_list and token.lemma_.lower() not in [verb.lemma_.lower() for verb in doc if verb.pos_ == 'VERB'] and token.lemma_.lower() not in spanish_verbs]
        for token in doc:
            if token.pos_ == 'NOUN' and token.text.lower() not in black_list:
                if token.lemma_.lower() not in glocary and token.text.lower() not in glocary:
                    nouns.append(token.lemma_)
        
        for word in original_text.split():
            if word.isupper() and len(word) > 1:
                word = word.replace(".", "")
                nouns.append(word)

        lowercase_list = [word.lower().replace(".","").replace(",","") for word in nouns]
        for i in brand_list:
            st.write('Brand is: ',i)
            if i.lower() in text.split() and i.lower() not in lowercase_list:
                st.write('adding brand')
                nouns.append(i)

        return nouns
        
    def count_occurrences(nouns):
        noun_counts = Counter(nouns)
        return noun_counts
    
    original_text = TranscriptText
    spanish_text = TranscriptText.lower()
    nouns = extract_nouns(spanish_text,original_text)
    noun_occurrences = count_occurrences(nouns)
    return noun_occurrences
    
def delete_all_files_in_directory(directory_path):
    try:
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # Check if it is a file (not a directory) and delete it
            if os.path.isfile(file_path):
                os.remove(file_path)
        st.toast(f"All stored files deleted successfully.", icon='😍')

    except Exception as e:
        st.write(f"An error occurred: {e}")

def detect_silence(path, time):
    print('inside')
    # Full path to the ffmpeg executable
    # ffmpeg_path = '/usr/bin/ffmpeg'  # Replace this with the actual path on your system

    # Use os.path.join to create the full command
    # command = f'{ffmpeg_path} -i {path} -af silencedetect=n=-17dB:d={str(time)} -f null -'
    command = [
        '/usr/bin/ffmpeg',
        '-i', path,
        '-af', f'silencedetect=n=-10dB:d={str(time)}',
        '-f', 'null', '-'
    ]
    # st.write(command)
    try:
        out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
    except Exception as e:
        st.error(f"Error executing command: {e}")

    s = stdout.decode("utf-8")
    k = s.split('[silencedetect @')
    
    if len(k) == 1:
        # print(stderr)
        return None

    start, end = [], []
    for i in range(1, len(k)):
        x = k[i].split(']')[1]
        if i % 2 == 0:
            x = x.split('|')[0]
            x = x.split(':')[1].strip()
            # print(x)
            end.append(float(x))
        else:
            x = x.split(':')[1]
            x = x.split('size')[0]
            x = x.replace('\r', '')
            x = x.replace('\n', '').strip()
            
            try:
                start.append(float(x))
            except ValueError as e:
                x=x.split('[')[0]
                start.append(float(x))
    # print(list(zip(start, end)))
    return list(zip(start, end))

def convert_audio_to_text(input_path,output_dir,similarity_brands,replacement_words,brand_list,max_size_mb=25):
    with st.spinner('converting audio to the standard format'):
        # AudioSegment.converter = "ffmpeg/bin/ffmpeg.exe"                  
        # utils.get_prober_name = get_prober_name
        audio = AudioSegment.from_file(input_path)
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path.name))[0] + ".wav")
        audio.export(output_path, format="wav")
    
    with st.spinner('splitting audio'):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB       
        if file_size_mb <= max_size_mb:
            parts = [output_path]  

        else:
            # print('under else')
            parts = []
            segment_size_ms = 100000  # Adjust this value based on your needs (10 seconds in this example)
    
            for start in range(0, len(audio), segment_size_ms):
                end = start + segment_size_ms
                part = audio[start:end]
                part_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path.name))[0]}_{start}-{end}.wav")
                part.export(part_path, format="wav")
                parts.append(part_path)
  
    # audio_duration = len(audio) / 1000
    with st.spinner(f'extracting text from {len(parts)} audio files'):
        text = ""
        for i in parts:
    
            audio_file = open(i, "rb")
            transcript = client.audio.transcriptions.create(
            model="whisper-1",
            prompt = f"Al convertir un audio en texto, asegúrese de escribir correctamente los nombres de las marcas. Estos son algunos nombres de marcas:{brand_list}",
            file=audio_file
            )
            text += transcript.text + " "

        for similar, replace in zip(similarity_brands, replacement_words):
            pattern = re.compile(re.escape(similar), re.IGNORECASE)
            text = pattern.sub(replace, text)

        return text,output_path

def main():
    try:
        st.title("Audio to Text Conversion App")
        if "audio_file" not in st.session_state:
            st.session_state.audio_file = None
        uploaded_file1 = st.sidebar.file_uploader("Upload dictionary.xlsx file", type=["xlsx"])
        if uploaded_file1 is not None:
            with open("dictionary.xlsx", "wb") as f:
                f.write(uploaded_file1.getbuffer())

        uploaded_file2 = st.sidebar.file_uploader("Upload glosary.xlsx file", type=["xlsx"])
        if uploaded_file2 is not None:
            with open("glosary.xlsx", "wb") as f:
                f.write(uploaded_file2.getbuffer())
                
        excel_file_path = 'dictionary.xlsx'

        df = pd.read_excel(excel_file_path)
        column_lists = [df[column].dropna().tolist() for column in df.columns]
        black_list = column_lists[0]
        brand_list = column_lists[1]
        skip_nouns = column_lists[2]
        similarity_brands = column_lists[3]
        replacement_words = column_lists[4]
        
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
        if uploaded_file is not None:
            st.session_state.audio_file = uploaded_file
            st.audio(st.session_state.audio_file, format="audio/wav", start_time=0)

            if st.button("Convert to Text"):
                TranscriptText,output_path = convert_audio_to_text(st.session_state.audio_file,'audios',similarity_brands,replacement_words,brand_list)
                text,duration,nouns,details = st.tabs(["Audio To Text","Duration","Nouns","Detail"])
        
                with text:
                    st.write(TranscriptText)
                
                with duration:
                    # st.write("PATH:", os.environ["PATH"])
                    # ffmpeg_path = shutil.which("ffmpeg")
                    # st.write("ffmpeg path:", ffmpeg_path)
                    # st.write("Current Working Directory:", os.getcwd())
                    silence_list = detect_silence(output_path, time = 4)
                    # st.write(silence_list)
                    if silence_list is not None:
                        start,end = silence_list[0]
                        start1,end1 = silence_list[-1]
                        audio_start = math.ceil(end)
                        # audio_end = round(end1 / 60, 1)
                        # st.write(audio_length)
                        end_minutes, end_seconds = divmod(end1, 60)
                        audio_duration = end1 - audio_start
                        st.text('Audio Details:')
                        duration_minutes, duration_seconds = divmod(audio_duration, 60)
                        st.text(f"Start: {audio_start}sec")
                        st.text(f"End: {int(end_minutes)} min {int(end_seconds)} sec")
                        st.text(f"Audio Duration: {int(duration_minutes)} min {int(duration_seconds)} sec")
                    else:
                        st.text('No Silence Found')
                with st.spinner('Extracting Nouns'):    
                    with nouns:
                        ## excel_file_path = 'dictionary.xlsx'
                        # df = pd.read_excel(excel_file_path)
                        # column_lists = [df[column].dropna().tolist() for column in df.columns]
                        # black_list = column_lists[0]
                        # brand_list = column_lists[1]
                        # skip_nouns = column_lists[2]
                        # similarity_brands = column_lists[3]
                        # replacement_words = column_lists[4]

                        glosary_file_path = 'glosary.xlsx'
                        df1 = pd.read_excel(glosary_file_path,header=None)
                        glocary = [word for col in df1.columns for word in df1[col].dropna()]
                        
                        noun_occurrences = extract_nouns_with_counts(glocary,black_list,TranscriptText,brand_list)
                        chars_to_remove = [',', '.', '...']
                        translator = str.maketrans('', '', ''.join(chars_to_remove))

                        cleaned_paragraph = TranscriptText.lower().translate(translator)
                        paragraph_words = cleaned_paragraph.split()  

                        for noun, count in noun_occurrences.items():
                            if noun.lower() in brand_list:
                                st.write(f"Noun: {noun}, Occurrences: {paragraph_words.count(noun.lower())}")
                            elif noun.lower() in skip_nouns:
                                continue
                            elif noun.lower() in similarity_brands:

                                replacement_word_index = similarity_brands.index(noun.lower())
                                replacement_word = replacement_words[replacement_word_index]

                                st.write(f"Noun: {replacement_word}, Occurrences: {paragraph_words.count(noun.lower())}")
                            else:
                                st.write(f"Noun: {noun}, Occurrences: {count}")

                with st.spinner('analysing the conversation to fetch the required details'):    
                    with details:
                        assistant = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        temperature=0,
                        messages=[
                            {"role": "system", "content":'''I want to spend 200k€ in this approach.You need to give 4 details by checking the below text extracted 
from an audio in spanish and in the audio there will be 2 people one is client
and the other one is salesman. 
      
The first detail is "Concept", you need to understand the concept spoken in the audio.
For example customer talk about buying a car and talk about speed, 
engine, seat, security, etc so you need to understand the converation and 
which part is said by who (client & salesman). 
      
The second detail is "Client Profile", you need to identify the client profile based on
the following preferences of purchasing: (Trend,Money,Confort, Fidelity,Security and Pride). 

The third detail is "Distribution",you need to include a percentage distribution of the 
6 preferences of client profile for purchaising(Trend,Money,Confort, Fidelity,Security and Pride). 
And 'Pride' and 'Status' is same thing so use one of them. Make sure to mention all preferences and if
there is any preference that is not mentioned just mention 0 against that preference name but mention all of the preference just for once.
      
And the fourth detail is to called "Probablity", you need to analyze the conversation based on client profile 
and establish a probability of purchasing from 0 to 1 and just give one number let suppose if it is 0.9 just mention 0.9 not 90% and give description and while giving description not use a word "could be" you must be sure.

In the output each detail should be in the new line like this:
Concept:
Client Profile:
Distribution:
Probablity:

Note: Make sure answer will be in spanish'''},
                            {"role": "user", "content":TranscriptText}]
                        )
                        result = assistant.choices[0].message.content

                        for similar, replace in zip(similarity_brands, replacement_words):
                            pattern = re.compile(re.escape(similar), re.IGNORECASE)
                            result = pattern.sub(replace, result)

                        st.write(result)

                delete_all_files_in_directory('audios')
    except Exception as e:
        st.write(e)         
                
if __name__ == "__main__":
    main()
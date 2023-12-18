import streamlit as st
import speech_recognition as sr
from pydub import utils,AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess
import math
import ffmpeg
import shutil
import spacy
from collections import Counter

load_dotenv()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_prober_name():
#     return "ffmpeg/bin/ffprobe.exe"

def extract_nouns_with_counts(TranscriptText):
    nlp = spacy.load('es_core_news_sm')
    spanish_verbs = ['hablar','cosa','hecho','gracia','gracias','solo','adiós','vale','respecto','poder','cuenta' ,'comer','vivir', 'trabajar', 'pensar', 'ser', 'estar', 'tener', 'poder', 'hacer', 'decir', 'ir', 'venir', 'saber', 'querer', 'dar', 'ver', 'sentir', 'oír', 'conocer', 'poner', 'salir', 'traer', 'caber', 'valer', 'conseguir', 'poder', 'poner', 'andar', 'caminar', 'correr', 'saltar', 'nadar', 'bailar', 'cantar', 'tocar', 'escribir', 'leer', 'estudiar', 'aprender', 'enseñar', 'viajar', 'conducir', 'pintar', 'cocinar',
                     'indicativo',	'presente',	'pretérito perfecto compuesto',	'tardo',	'tardas',	'tarda',	'tardamos',	'tardáis',	'tardan',	'futuro',	'tardaré',	'tardarás',	'tardará',	'tardaremos',	'tardaréis',	'tardarán',	'pretérito imperfecto',	'tardaba',	'tardabas',	'tardaba',	'tardábamos',	'tardabais',	'tardaban',	'publicidad',	'he tardado',	'has tardado',	'ha tardado',	'hemos tardado',	'habéis tardado',	'han tardado',	'pretérito pluscuamperfecto',	'había tardado',	'habías tardado',	'había tardado',	'habíamos tardado',	'habíais tardado',	'habían tardado',	'pretérito anterior',	'hube tardado',	'hubiste tardado',	'hubo tardado',	'hubimos tardado',	'hubisteis tardado',	'hubieron tardado',	'futuro perfecto',	'habré tardado',	'habrás tardado',	'habrá tardado',	'habremos tardado',	'habréis tardado',	'habrán tardado',	'condicional perfecto',	'habría tardado',	'habrías tardado',	'habría tardado',	'habríamos tardado',	'habríais tardado',	'habrían tardado',	'condicional',	'tardaría',	'tardarías',	'tardaría',	'tardaríamos',	'tardaríais',	'tardarían',	'pretérito perfecto simple',	'tardé',	'tardaste',	'tardó',	'tardamos',	'tardasteis',	'tardaron',	'imperativo',	'tarda',	'tarde',	'tardemos',	'tardad',	'tarden',	'subjuntivo',	'presente',	'tarde',	'tardes',	'tarde',	'tardemos',	'tardéis',	'tarden',	'futuro',	'tardare',	'tardares',	'tardare',	'tardáremos',	'tardareis',	'tardaren',	'pretérito imperfecto',	'tardara',	'tardaras',	'tardara',	'tardáramos',	'tardarais',	'tardaran',	'pretérito pluscuamperfecto',	'hubiera tardado',	'hubieras tardado',	'hubiera tardado',	'hubiéramos tardado',	'hubierais tardado',	'hubieran tardado',	'futuro perfecto',	'hubiere tardado',	'hubieres tardado',	'hubiere tardado',	'hubiéremos tardado',	'hubiereis tardado',	'hubieren tardado',	'pretérito imperfecto (2)',	'tardase',	'tardases',	'tardase',	'tardásemos',	'tardaseis',	'tardasen',	'pretérito pluscuamperfecto (2)',	'hubiese tardado',	'hubieses tardado',	'hubiese tardado',	'hubiésemos tardado',	'hubieseis tardado',	'hubiesen tardado',	'pretérito perfecto',	'haya tardado',	'hayas tardado',	'haya tardado',	'hayamos tardado',	'hayáis tardado',	'hayan tardado',	'gerundio',	'tardando',	'gerundio compuesto',	'habiendo tardado',	'infinitivo',	'tardar',	'infinitivo compuesto',	'haber tardado',	'participio pasado',	'tardado',
                     'indicativo',	'presente',	'pretérito perfecto compuesto',	'soy',	'eres',	'es',	'somos',	'sois',	'son',	'futuro',	'seré',	'serás',	'será',	'seremos',	'seréis',	'serán',	'pretérito imperfecto',	'era',	'eras',	'era',	'éramos',	'erais',	'eran',	'publicidad',	'he sido',	'has sido',	'ha sido',	'hemos sido',	'habéis sido',	'han sido',	'pretérito pluscuamperfecto',	'había sido',	'habías sido',	'había sido',	'habíamos sido',	'habíais sido',	'habían sido',	'pretérito anterior',	'hube sido',	'hubiste sido',	'hubo sido',	'hubimos sido',	'hubisteis sido',	'hubieron sido',	'futuro perfecto',	'habré sido',	'habrás sido',	'habrá sido',	'habremos sido',	'habréis sido',	'habrán sido',	'condicional perfecto',	'habría sido',	'habrías sido',	'habría sido',	'habríamos sido',	'habríais sido',	'habrían sido',	'condicional',	'sería',	'serías',	'sería',	'seríamos',	'seríais',	'serían',	'pretérito perfecto simple',	'fui',	'fuiste',	'fue',	'fuimos',	'fuisteis',	'fueron',	'imperativo',	'sé',	'sea',	'seamos',	'sed',	'sean',	'subjuntivo',	'presente',	'sea',	'seas',	'sea',	'seamos',	'seáis',	'sean',	'futuro',	'fuere',	'fueres',	'fuere',	'fuéremos',	'fuereis',	'fueren',	'pretérito imperfecto',	'fuera',	'fueras',	'fuera',	'fuéramos',	'fuerais',	'fueran',	'pretérito pluscuamperfecto',	'hubiera sido',	'hubieras sido',	'hubiera sido',	'hubiéramos sido',	'hubierais sido',	'hubieran sido',	'futuro perfecto',	'hubiere sido',	'hubieres sido',	'hubiere sido',	'hubiéremos sido',	'hubiereis sido',	'hubieren sido',	'pretérito imperfecto (2)',	'fuese',	'fueses',	'fuese',	'fuésemos',	'fueseis',	'fuesen',	'pretérito pluscuamperfecto (2)',	'hubiese sido',	'hubieses sido',	'hubiese sido',	'hubiésemos sido',	'hubieseis sido',	'hubiesen sido',	'pretérito perfecto',	'haya sido',	'hayas sido',	'haya sido',	'hayamos sido',	'hayáis sido',	'hayan sido',	'gerundio',	'siendo',	'gerundio compuesto',	'habiendo sido',	'infinitivo',	'ser',	'infinitivo compuesto',	'haber sido',	'participio pasado',	'sido',
                     'indicativo',	'presente',	'pretérito perfecto compuesto',	'tardo',	'tardas',	'tarda',	'tardamos',	'tardáis',	'tardan',	'futuro',	'tardaré',	'tardarás',	'tardará',	'tardaremos',	'tardaréis',	'tardarán',	'pretérito imperfecto',	'tardaba',	'tardabas',	'tardaba',	'tardábamos',	'tardabais',	'tardaban',	'publicidad',	'he tardado',	'has tardado',	'ha tardado',	'hemos tardado',	'habéis tardado',	'han tardado',	'pretérito pluscuamperfecto',	'había tardado',	'habías tardado',	'había tardado',	'habíamos tardado',	'habíais tardado',	'habían tardado',	'pretérito anterior',	'hube tardado',	'hubiste tardado',	'hubo tardado',	'hubimos tardado',	'hubisteis tardado',	'hubieron tardado',	'futuro perfecto',	'habré tardado',	'habrás tardado',	'habrá tardado',	'habremos tardado',	'habréis tardado',	'habrán tardado',	'condicional perfecto',	'habría tardado',	'habrías tardado',	'habría tardado',	'habríamos tardado',	'habríais tardado',	'habrían tardado',	'condicional',	'tardaría',	'tardarías',	'tardaría',	'tardaríamos',	'tardaríais',	'tardarían',	'pretérito perfecto simple',	'tardé',	'tardaste',	'tardó',	'tardamos',	'tardasteis',	'tardaron',	'imperativo',	'tarda',	'tarde',	'tardemos',	'tardad',	'tarden',	'subjuntivo',	'presente',	'tarde',	'tardes',	'tarde',	'tardemos',	'tardéis',	'tarden',	'futuro',	'tardare',	'tardares',	'tardare',	'tardáremos',	'tardareis',	'tardaren',	'pretérito imperfecto',	'tardara',	'tardaras',	'tardara',	'tardáramos',	'tardarais',	'tardaran',	'pretérito pluscuamperfecto',	'hubiera tardado',	'hubieras tardado',	'hubiera tardado',	'hubiéramos tardado',	'hubierais tardado',	'hubieran tardado',	'futuro perfecto',	'hubiere tardado',	'hubieres tardado',	'hubiere tardado',	'hubiéremos tardado',	'hubiereis tardado',	'hubieren tardado',	'pretérito imperfecto (2)',	'tardase',	'tardases',	'tardase',	'tardásemos',	'tardaseis',	'tardasen',	'pretérito pluscuamperfecto (2)',	'hubiese tardado',	'hubieses tardado',	'hubiese tardado',	'hubiésemos tardado',	'hubieseis tardado',	'hubiesen tardado',	'pretérito perfecto',	'haya tardado',	'hayas tardado',	'haya tardado',	'hayamos tardado',	'hayáis tardado',	'hayan tardado',	'gerundio',	'tardando',	'gerundio compuesto',	'habiendo tardado',	'infinitivo',	'tardar',	'infinitivo compuesto',	'haber tardado',	'participio pasado',	'tardado']
    def extract_nouns(text):
        doc = nlp(text)
        nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and token.lemma_ not in [verb.lemma_ for verb in doc if verb.pos_ == 'VERB'] and token.lemma_ not in spanish_verbs]
        return nouns

    def count_occurrences(nouns):
        noun_counts = Counter(nouns)
        return noun_counts
    
    spanish_text = TranscriptText.lower()
    nouns = extract_nouns(spanish_text)
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

# def detect_silence(path, time):
#     print('inside')
#     # Full path to the ffmpeg executable
#     ffmpeg_path = '/usr/bin/ffmpeg'  # Replace this with the actual path on your system

#     # Use os.path.join to create the full command
#     command = f'{ffmpeg_path} -i {path} -af silencedetect=n=-17dB:d={str(time)} -f null -'
    
#     out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     stdout, stderr = out.communicate()
#     s = stdout.decode("utf-8")
#     k = s.split('[silencedetect @')
    
#     if len(k) == 1:
#         # print(stderr)
#         return None

#     start, end = [], []
#     for i in range(1, len(k)):
#         x = k[i].split(']')[1]
#         if i % 2 == 0:
#             x = x.split('|')[0]
#             x = x.split(':')[1].strip()
#             # print(x)
#             end.append(float(x))
#         else:
#             x = x.split(':')[1]
#             x = x.split('size')[0]
#             x = x.replace('\r', '')
#             x = x.replace('\n', '').strip()
            
#             try:
#                 start.append(float(x))
#             except ValueError as e:
#                 x=x.split('[')[0]
#                 start.append(float(x))
#     # print(list(zip(start, end)))
#     return list(zip(start, end))

def convert_audio_to_text(input_path,output_dir,max_size_mb=25):
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
  
    audio_duration = round(len(audio) / 60000, 1)
    with st.spinner(f'extracting text from {len(parts)} audio files'):
        text = ""
        for i in parts:
    
            audio_file = open(i, "rb")
            transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
            )
            text += transcript.text + " "

        return text,audio_duration,output_path

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
                TranscriptText,audio_length,output_path = convert_audio_to_text(st.session_state.audio_file,'audios')
                # converted_files  = convert_audio_to_text(st.session_state.audio_file,'audios')
                # print("Converted and/or split files:", converted_files)
                # Add tabs
                text,duration,nouns,details = st.tabs(["Audio To Text","Duration","Nouns","Detail"])
        
                with text:
                    # for utterance in transcript.utterances:
                    #     st.write(f"Speaker {utterance.speaker}: {utterance.text}")
                    st.write(TranscriptText)
                
                with duration:
                    # st.write("PATH:", os.environ["PATH"])
                    
                    # ffmpeg_path = shutil.which("ffmpeg")
                    # st.write("ffmpeg path:", ffmpeg_path)
                    # st.write("Current Working Directory:", os.getcwd())
                    # silence_list = detect_silence(output_path, time = 4)
                    # start,end = silence_list[0]
                    # start1,end1 = silence_list[-1]
                    # audio_start = math.floor(end)
                    # audio_end = round(end1 / 60, 1)

                    # audio_duration = audio_length - round(audio_start / 60, 1)

                    # st.write('Audio Details:')
                    # st.write(f"Start: {audio_start}sec")
                    # st.write(f"End: {audio_end}min(s)")
                    # st.write(f"Audio Duration: {audio_duration}min(s)")
                
                    st.write('In progress')
                with st.spinner('Extracting Nouns'):    
                    with nouns:
                        noun_occurrences = extract_nouns_with_counts(TranscriptText)
                        for noun, count in noun_occurrences.items():
                            st.write(f"Noun: {noun}, Occurrences: {count}")

                with st.spinner('analysing the conversation to fetch the required details'):    
                    with details:
                        assistant = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        temperature=0,
                        messages=[
                            {"role": "system", "content":'''You need to give 4 details by checking the below text extracted 
                        from an audio in spanish and in the audio there will be 2 people one is client
                        and the other one is salesman. 
                             
                        The first detail is "Concept", you need to understand the concept spoken in the audio.
                        For example customer talk about buying a car and talk about speed, 
                        engine, seat, security, etc so you need to understand the converation and 
                        which part is said by who (client & salesman). 
                             
                        The second detail is "Client Profile", you need to identify the client profile based on
                        the following preferences of purchasing: (Trend,Money,Confort, Fidelity,Security and Pride). 
                        
                        The third detail is "Distribution",you need to include a percentage distribution of the 
                        6 preferences of client profile for purchaising. And 'Pride' and 'Status' is same thing so use one of them.
                             
                        And the fourth detail is to called "Probablity", you need to analyze the conversation based on client profile 
                        and establish a probability of purchasing from 0 to 1.

                        In the output each detail should be in the new line like this:
                        Concept:
                        Client Profile:
                        Distribution:
                        Probablity:
                        
                        Note: Make sure answer will be in spanish'''},
                        {"role": "user", "content":TranscriptText}]
                        )
                        result = assistant.choices[0].message.content
                        st.write(result)

                delete_all_files_in_directory('audios')
    except Exception as e:
        st.write(e)         
                
if __name__ == "__main__":
    main()

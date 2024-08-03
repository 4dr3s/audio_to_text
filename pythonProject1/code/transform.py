# Python 3.9
import moviepy.editor as mp
from huggingsound import SpeechRecognitionModel
import soundfile as sf
from scipy.signal import resample
import pandas as pd
import json
file_path = "C:/Users/andre/Downloads/datos_tiktok_threads_instagram_twitter.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.json_normalize(data)


def change_description(description):
    return str(description)


path = 'C:/Universidad/Legislacion/videos/'
for i, item in df.iterrows():
    try:
        if item['fuente'] == 'Tik Tok':
            video_path = path + f"{item['id']}.mp4"
            video_file = mp.VideoFileClip(video_path)
            audio_path = path + f"{item['id']}.mp3"
            video_file.audio.write_audiofile(audio_path)
            data, sample_rate = sf.read(audio_path)
            if sample_rate != 16000:
                data = resample(data, int(len(data) * 16000 / sample_rate))
            sf.write(audio_path, data, 16000)
            model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
            audio_paths = [audio_path]
            transcriptions = model.transcribe(audio_paths)
            if transcriptions:
                transcription_text = transcriptions[0]['transcription']
                df.at[i, "description"] = change_description(transcription_text)
    except Exception as ex:
        print(ex)
        continue

modified_data = df.to_dict(orient='records')
output_file_path = "C:/Users/andre/Downloads/datos_tiktok_threads_instagram_twitter_get_video_description.json"
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(modified_data, f, ensure_ascii=False, indent=4)
import librosa as lr
import numpy as np
import soundfile as sf


def audio_from_file(path, sr=48000, offset=0.0, duration=None):
    try:
        audio, _ = lr.load(
            path,
            sr=sr,
            mono=True,
            offset=offset,
            duration=duration,
            dtype=np.float32,
            res_type="kaiser_fast",
        )

        lr.mu_expand(audio)
        print(audio[:5])
        # audio.shape = (-1, 1)
        return audio
    except ValueError as ex:
        print("value error {}\n{}".format(path, ex))
        return []
    except Exception as ex:
        print("could not read {}\n{}".format(path, ex))
        return None


if __name__ == "__main__":
    file_name = "sample-audio/Record_mulaw-3.wav"
    audio_array = audio_from_file(file_name)
    audio_array.shape = (-1, 1)
    print(audio_array.shape)
    sf.write("sound.wav", audio_array, 48000)

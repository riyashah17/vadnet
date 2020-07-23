import audioop
import base64
import pickle
import numpy as np
from flask import Flask, request
from librosa import to_mono, resample, mu_compress, mu_expand
from vadnet.predict_audio import Predictor
import json
import time
import soundfile

predictor = Predictor()
app = Flask(__name__)
sr_native = 8000
sr_target = 48000


@app.route("/predict", methods=["POST"])
def api_message():
    if request.headers["Content-Type"] == "application/json":
        now = time.time()
        data = request.data
        request_read_time = time.time() - now
        now = time.time()

        audio_dict = json.loads(data)
        audio_data = audio_dict['msg']

        # remove the WAVE header - note - not do be done in twilio integration
        message_bytes = base64.b64decode(audio_data)
        audio_array = np.frombuffer(audioop.ulaw2lin(message_bytes, 2), dtype=np.int16) / 32767
        audio_array = resample(audio_array, sr_native, sr_target, res_type="kaiser_fast")
        audio_array.shape = (-1, 1)
        print(audio_array.shape)

        if isinstance(audio_array, (list, tuple)):
            audio_array, granularity = audio_array
        else:
            audio_array, granularity = audio_array, None
        buffer_read_time = time.time() - now
        now = time.time()
        result = predictor.run(audio_array, granularity=granularity)
        prediction_time = time.time() - now
        return json.dumps({
            "result": [i.tolist() for i in result],
            "prediction_time": prediction_time,
            "buffer_read_time": buffer_read_time,
            "request_read_time": request_read_time,
            "granularity": 1 if granularity is None else granularity
        })
    else:
        return "415 Unsupported Media Type"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

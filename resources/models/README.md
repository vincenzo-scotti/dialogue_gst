# Models

This directory is used to host the pre-trained model directories.
These directories contain both models and tokenisers trained during the experiments.
It is possible to download a zip archive with all the trained models at [this link]().

GST prediction:
- [DGST]()
  - Using Therapy-DLDLM response embedding computed from context ([link](https://polimi365-my.sharepoint.com/:u:/g/personal/10451445_polimi_it/EYvTr-aD0glErLSBhKf1J18BgetFIAhC_MO1iugkFHwrhg?e=2ReQdb)).

For the checkpoints of the DGST models, refer to `./experiments/README.md`.

Response generation:
- [DialoGPT](https://aclanthology.org/2020.acl-demos.30/):
  - Small ([link](https://huggingface.co/microsoft/DialoGPT-small));
  - Medium ([link](https://huggingface.co/microsoft/DialoGPT-medium));
  - Large ([link](https://huggingface.co/microsoft/DialoGPT-large));
- Therapy-DLDLM ([link](https://polimi365-my.sharepoint.com/:u:/g/personal/10451445_polimi_it/EQ7PspwlveNPnXsB4Bl7T2wBxpa6SGVS3hTaBAEvFatTWA?e=qC8CxS)).
  
Please refer to the [Transformers](https://huggingface.co/docs/transformers/index) library by [HuggingFace](https://huggingface.co) for further details on the DialoGPT language models.

Voice synthesis:
- Spectrogram generation:
  - [Mellotron](https://doi.org/10.1109/ICASSP40776.2020.9054556) with Speaker conditioning, pitch (F0) conditioning and GST conditioning:
    - Trained on [LibriTTS](https://openslr.org/60/) ([weights checkpoint](https://drive.google.com/open?id=1ZesPPyRRKloltRIuRnGZ2LIUEuMSVjkI));
    - Trained on [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) ([weights checkpoint](https://drive.google.com/open?id=1UwDARlUl8JvB2xSuyMFHFsIWELVpgQD4)).  
  - [Tacotron 2](https://doi.org/10.1109/ICASSP.2018.8461368) ([weights checkpoint](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing))
- Vocoder:
  - [WaveGlow](https://doi.org/10.1109/ICASSP.2019.8683143) ([weights checkpoint](https://drive.google.com/open?id=1okuUstGoBe_qZ4qUEF8CcwEugHP7GM_b))
  
Please refer to the [Mellotron API](https://github.com/vincenzo-scotti/tts_mellotron_api) we developed for the credits on the original work to develop the TTS and for further references and details on the TTS models.
For simplicity we provide a separate zip file with all the model checkpoints necessary to speech synthesis ([link](https://polimi365-my.sharepoint.com/:u:/g/personal/10451445_polimi_it/Eb5jr0ERxy5MuWIZopg3iwYBuq8D8IzFZJdzLN8f4bSEcA?e=wtI0dz)).

Directory structure:
```
 |- models/
    |- dgst/
      |- dgst.pt
    |- tts/
      |- mellotron/
        |- mellotron_ljs.pt
        |- mellotron_libritts.pt
      |- tacotron_2
        |- tacotron2_statedict.pt
    |- vocoder
      |- waveglow/
        |- waveglow_256channels_universal_v4.pt
```

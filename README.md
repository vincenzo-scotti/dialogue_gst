# Dialogue GST

Codebase for the extended summary "[On the Role of Dialogue Context in Predicting Speaking Style](https://www.overleaf.com/read/cjqcjkkxntkp)". 
This repository contains the implementation of the DialogueGST described in the paper.

## Repository structure

This repository is organised into four main directories:

- `experiments/` contains the directories to host:  
    - results of the experiments;
    - checkpoints generated during the experiments;
    - experiment configuration dumps;
    - experiment logs.
- `notebooks/` contains the directories to host:
    - usage example notebook.
- `resources/` contains:
    - directories to host the dialogue corpora used in the experiments, and the references to download them;
    - directory to host the YAML configuration files to run the experiments.
    - directory to host the pre-trained models, and the references to download them.
- `src/` contains modules and scripts to: 
    - run training and evaluation steps;
    - load and preprocess corpora.

For further details, refer to the `README.md` within each directory.

## Environment

To install all the required packages within an anaconda environment, run the following commands:

```bash
# Create anaconda environment (skip cudatoolkit option if you don't want to use the GPU)
conda create -n dgst python=3.10 cudatoolkit=11.3
# Activate anaconda environment
conda activate dgst
# Install packages
conda install pytorch=1.11.0 -c pytorch
conda install -c conda-forge numpy=1.21 transformers tensorboard pandas scikit-learn librosa matplotlib seaborn jupyterlab
# Additional packages
conda install -c conda-forge scipy=1.9.1 tensorflow music21 inflect tensorboardx unidecode pydantic=1.10.2
conda install -c anaconda nltk pillow
pip install jamo
# Download and initialise TTS API submodule
git submodule init; git submodule update
# NOTE follow the API instructions to complete installation
```

To add the directories to the Python path, you can add these lines to the file `~/.bashrc`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/dialoguegst/src
export PYTHONPATH=$PYTHONPATH:/path/to/dialoguegst/submodules/tts_mellotron_api/src
export PYTHONPATH=$PYTHONPATH:/path/to/dialoguegst/submodules/tts_mellotron_api/submodules
export PYTHONPATH=$PYTHONPATH:/path/to/dialoguegst/submodules/tts_mellotron_api/submodules/mellotron
export PYTHONPATH=$PYTHONPATH:/path/to/dialoguegst/submodules/tts_mellotron_api/submodules/mellotron/waveglow
export PYTHONPATH=$PYTHONPATH:/path/to/dialoguegst/submodules/tts_mellotron_api/submodules/tacotron2
```

## Training

### Run

There is a script to train or fine-tune the model, it expects to have `./src` in the Python path and all data sets to be downloaded and placed in the `./resources/data/raw/` directory.

To train the model run:
```bash
python ./src/bin/train_dialogue_gst.py --config_file_path ./resources/configs/path/to/config.yaml
```

To train the model in background run:

```bash
nohup python ./src/bin/train_dialogue_gst.py --config_file_path ./resources/configs/path/to/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
```

### Monitor

To connect to a remote server and monitor the training process via [Tensorboard](https://www.tensorflow.org/tensorboard) connect via ssh to your machine using a tunnel

```bash
ssh  -L 16006:127.0.0.1:6006 user@adderess
```

Start the Tensorboard server on the remote machine

```bash
tensorboard --logdir ./expertiments/path/to/tensorboard/
```

Finally, connect to http://127.0.0.1:16006 on your local machine

## References

If you are willing to use our code or our models, please cite our work through the following BibTeX entry:
```bibtex

```

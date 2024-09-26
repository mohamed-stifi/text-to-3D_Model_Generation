
# Text to 3D Model Generation

This repository provides an implementation for GET3D with textual conditionning for converting text input into 3D models. 
This project is divided project package in src `textTo3DGen` and multiple pipelines, each pipline stage is designed to handle a specific part of the process, from data ingestion to model training. Additionally, each stage includes a Jupyter notebook in the `research` folder that explains how thius pipeline works in detail.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Pipelines](#pipelines)
  - [Data Ingestion](#data-ingestion)
  - [Data Rendering](#data-rendering)
  - [Text Embedding](#text-embedding)
  - [Data Split](#data-split)
  - [Training Model](#training-model)
- [Research Notebooks](#research-notebooks)
- [Logging](#logging)

## Installation

To use this repository, clone it and install the required dependencies. Make sure you have Python installed, then run:

```bash
git clone https://github.com/yourusername/textTo3DModelGen.git
cd textTo3DModelGen
pip install -r requirements.txt
```

## Usage

You can run different stages of the pipeline using command-line arguments. The available stages are:

```bash
python main.py --stage <stage_number>
```

### Available Stages

- `1`: Run the Data Ingestion pipeline.
- `2`: Run the Data Rendering pipeline.
- `3`: Run the Text Embedding pipeline.
- `4`: Run the Data Split pipeline.
- `5`: Run the Training Model pipeline.
- `6`: Run all pipelines (not implemented yet).

## Pipelines

### Data Ingestion

The Data Ingestion stage collects and prepares raw data for further processing.

- **Usage**: `python main.py --stage 1`
- **Notebook**: `research/01_data_ingestion.ipynb`

### Data Rendering

The Data Rendering stage generates visual representations of the data.

- **Usage**: `python main.py --stage 2`
- **Notebook**: `research/02_data_rendering.ipynb`

### Text Embedding

The Text Embedding stage converts text data into embeddings suitable for model training.

- **Usage**: `python main.py --stage 3`
- **Notebook**: `research/03_text_embedding.ipynb`

### Data Split

The Data Split stage divides the dataset into training and testing sets.

- **Usage**: `python main.py --stage 4`
- **Notebook**: `research/04_data_split.ipynb`

### Training Model

The Training Model stage trains the model using the prepared data.

- **Usage**: `python main.py --stage 5`
- **Notebook**: `research/05_training_model.ipynb`

## Research Notebooks

Each stage has a corresponding Jupyter notebook located in the `research` directory. These notebooks provide detailed explanations of the methodology, algorithms used, and implementation details.

## Logging

This repository uses a logging system to track the progress of each stage. Log messages are written to the console, providing feedback on the execution of each pipeline stage. Any errors encountered will also be logged for debugging purposes.

```python
from textTo3DModelGen import logger
```

To see the logs, ensure that your logging configuration is set up correctly.

# AI-Microstructure-Predictor

## Overview

This project focuses on predicting the microstructure outcomes of additively manufactured (AM) metals based on key process parameters such as laser power, scanning speed, and powder feed rate.

It combines classical machine learning models with early-stage deep learning prototypes to understand and optimize the material properties critical for advanced manufacturing.

This work is being carried out in collaboration with the **Singapore Centre for 3D Printing (SC3DP) at Nanyang Technological University (NTU)**, under the guidance of **Dr. Shubham Chandra**.

The goal is to assist researchers and engineers in accelerating the design of superior AM parts by linking process control to microstructural quality.

## Features

- Supervised learning models for microstructure classification and regression.
- Data preprocessing pipelines tailored for AM datasets.
- Modular codebase for easy experimentation with new algorithms.
- Notebook-driven experimentation for rapid iteration and visualization.

## Technologies Used

- Python 3.10+
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebooks

## Repository Structure

```
AI-Microstructure-Predictor/
├── data/           # Datasets for training/testing
├── notebooks/      # Jupyter notebooks for experimentation and EDA
├── src/            # Source code (model training, preprocessing, evaluation)
├── requirements.txt# Package requirements
├── LICENSE         # Open source license
├── README.md       # Project documentation
```

## Getting Started

1. **Clone the repository**
```bash
git clone https://github.com/Aryand43/AI-Microstructure-Predictor.git
cd AI-Microstructure-Predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run notebooks or scripts**
- Start experimenting with notebooks in `/notebooks`
- Train models using scripts in `/src`

## Future Development

- Integration of deep learning CNN models for feature extraction.
- Expansion to multi-phase material predictions.
- Real-time process-parameter recommendation systems.

## License

This project is licensed under the MIT License.

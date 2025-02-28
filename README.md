## Sentiment Analysis Using Recurrent Neural Networks: Binary Classification of Movie Reviews

Sentiment analysis in natural language processing (NLP) aims to classify text as positive, negative, or neutral.
Given the vast amount of textual data available, machine learning techniques have significantly advanced sentiment 
classification tasks, such as analyzing emails, social media posts, reviews, and customer support chats. Businesses 
use these models to monitor brand perception and detect customer opinion trends.

Recurrent Neural Networks (RNNs) have proven effective for processing sequential text structures, capturing 
long-range dependencies, and learning complex patterns. Unlike traditional models, they handle variable-length text 
without extensive preprocessing and leverage parallel computation for scalability.

This project explores NLP and RNN-based sentiment analysis, comparing simple and complex recurrent models for binary 
classification. It uses the IMDB dataset of 50,000 movie reviews, collected by Stanford University for binary sentiment 
classification. The dataset is preprocessed for balanced positive (≥7/10) and negative (≤4/10) labels, with training 
and test sets containing disjoint movies to prevent model bias. The implementation is done in Python using NumPy for 
data processing and TensorFlow Keras for model development and evaluation.

## Main Tools

- `Python` (Main programming language for the application) -- 3.11.5
- `NumPy` (Python library for numerical computing and array operations) -- 1.26.0
- `TensorFlow Keras` (Python deep learning API for building and training neural networks) -- 3.8.0

## Directory Structure

Here’s an overview of the project directory structure:

```plaintext
imdb-sentiment-analysis/
├── src/
│   ├── preprocess.py    							# Module for loading and preprocessing IMDB dataset
│   ├── models.py        							# Module for building RNN models
│   └── main.py          							# Main entry point for running the project and train the choosen model
├── .gitignore           							# For ignoring unnecessary files
├── LICENSE              							# MIT License
├── README.md            							# Project overview and setup instructions
├── requirements         							# List of python dependencies required for the project
├── statistics_bachelor_thesis_gph_2023_07_21.pdf	# Full Bachelor's thesis document, detailing the research, methodology and results (in catalan)
```

## Requirements

The following Python libraries are required to run this project:

*   `keras` -- 3.8.0
*   `matplotlib` -- 3.7.2
*   `numpy` -- 1.26.0

## Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/gapohi/imdb-sentiment-analysis.git
```

2. Navigate to the repository directory:
```bash
cd imdb-sentiment-analysis
```

3. Install the required libraries:
```bash
pip install -r requirements.txt
```

4. Run the main.py file
```bash
cd src
python main.py
```

### Model results

| Model                | Parameters | Accuracy |
|----------------------|------------|----------|
| Simple RNN           | 165,249    | 0.50     |
| Simple LSTM          | 85,281     | 0.70     |
| Simple GRU           | 175,809    | 0.50     |
| Bidirectional RNN    | 170,497    | 0.63     |
| Bidirectional LSTM   | 201,601    | 0.86     |
| Bidirectional GRU    | 191,617    | 0.86     |
| Deep RNN             | 181,761    | 0.50     |
| Deep LSTM            | 246,849    | 0.58     |
| Deep GRU             | 225,729    | 0.50     |

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would 
like to change. Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

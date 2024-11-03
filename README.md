# ASR Project Introduction and Objectives
## Introduction
        In the digital era, automatic speech recognition (ASR) has become a pivotal technology in voice-driven applications 
        like virtual assistants and customer support systems. As the demand for voice interfaces grows, achieving accurate 
        and efficient ASR is essential, especially for devices with limited resources. This project explores the integration 
        of pretrained ASR models, focusing on the Whisper-tiny model, to enhance speech recognition performance while 
        keeping resource requirements minimal.

        The Whisper-tiny model, developed by OpenAI, is known for its compact size and quick inference speed, making it 
        suitable for devices with restricted computational power. To test this model’s performance, we use the MINDS14 
        dataset by PolyAI, which includes diverse language and accent variations. This dataset offers a robust testing ground, 
        challenging the model to adapt to real-world speech nuances.

        Through this project, we aim to evaluate the effectiveness of the Whisper-tiny model in handling varied speech inputs 
        and to explore its potential for real-world applications in voice-driven systems.

## Objective

        The main objective of this research is to develop an ASR system tailored for the English language by using the 
        pretrained Whisper model. With the rich linguistic diversity of the MINDS14 dataset, we aim to:

        1. Improve speech recognition accuracy on devices with limited computational resources.
        2. Understand the strengths and limitations of pretrained ASR models in handling diverse language and accent variations.
        3. Evaluate the suitability of Whisper-tiny for deployment in resource-limited environments.

        This project seeks to advance ASR technology by showing the effectiveness of compact, efficient models in real-world 
        settings, making voice recognition solutions more accessible and adaptable.

## Loading and Merging the Dataset
The project uses the MINDS-14 dataset, specifically focusing on the US, Australian, and British English subsets, to assess the Whisper-tiny ASR model’s performance with varied English accents in the e-banking domain. The dataset is loaded through a structured MINDS14DataLoader class, which verifies the data’s integrity, checking for any missing or corrupted audio files and summarizing dataset statistics like sample count and intent distribution. The diversity of accents across these subsets helps train the model to better understand linguistic variations. After loading and verifying each subset, they are merged to increase the training data size, making the model more adaptable to different accents. This merging process includes checks for data loss, class balance, and consistency of features like transcriptions and audio properties. This approach aims to build a robust ASR model suited for real-world applications that encounter diverse English accents.

## Exploratory Data Analysis (EDA) Summary
1. Intent Class Distribution: A bar plot visualizes the frequency distribution of each intent class, providing insights into class balance. This ensures that the dataset covers a diverse range of intents, critical for effective model training.
2. Waveform and Spectrogram Analysis: For each intent, waveforms and spectrograms were generated to examine audio characteristics, such as amplitude changes and frequency components over time. This helps in understanding the variation in speech patterns across different intents.
3. Audio Feature Analysis with Chroma and FFT: Chroma (pitch class) and Fast Fourier Transform (FFT) analyses were conducted for selected intents. The chromagram reveals pitch variations, while FFT displays the magnitude spectrum, providing a detailed view of audio frequency components and aiding in feature extraction.
4. Null Value Check: The dataset was checked for null values in key features (e.g., audio, transcription, intent class) to ensure data completeness. No missing values were found, confirming the dataset's integrity.
5. Duplicate Analysis: Duplicate transcriptions were identified, helping to understand redundancy in data. While some duplication is acceptable, this analysis highlights repeated phrases that could potentially skew model training if overrepresented.

## Dataset Preprocessing
This preprocessing ensures that the data is clean, consistent, and suitable for training an ASR model, with a well-defined vocabulary and standardized audio format.

1. Dataset Splitting and Cleanup
- The dataset is divided into training and testing sets in an 80/20 ratio to ensure proper model evaluation.
- Unnecessary columns are removed to streamline the dataset for analysis.
- The structure of the dataset is verified to confirm that essential columns (e.g., transcription, intent class) are intact.
2. Transcription Verification
- Random samples of transcriptions are displayed to manually inspect the quality of text data.
- The audio and path columns are hidden to focus on transcription quality, helping identify any potential issues in text clarity or accuracy.
3. Text Cleaning
- Punctuation is removed, text is converted to lowercase, and extra spaces are standardized, ensuring uniformity across transcriptions.
- Samples of cleaned text are displayed to verify the effectiveness of the cleaning steps.

## Vocabulary Creation
# 6d. Create Vocabulary
```def extract_all_chars(batch):
    """
    Extract unique characters for vocabulary
    
    Args:
        batch: Dataset batch
    
    Returns:
        dict: Vocabulary and full text
    """
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

# Extract vocabulary from both train and test sets
vocabs = merged_dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=1,
    keep_in_memory=True,
    remove_columns=merged_dataset.column_names['train']
)

# Create vocabulary dictionary
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
print(vocab_dict)

# Replace space with pipe symbol
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# Add special tokens
vocab_dict["[UNK]"] = len(vocab_dict)  # Unknown token
vocab_dict["[PAD]"] = len(vocab_dict)  # Padding token
print(len(vocab_dict))
```

### Audio Standardization
- Since the MINDS-14 dataset uses an 8 kHz sampling rate, the audio files are converted to a 16 kHz sampling rate to match the input requirements of the Whisper model.
- The original sampling rate of each audio file is checked, then converted, and the result is verified to ensure consistency across the dataset.


### Dataset Extraction and Processor Testing
This extraction and processing pipeline ensures that the dataset is formatted and ready for ASR model training, with all audio and text data appropriately encoded for model input.

#### Processor Testing
1. Component Initialization: The smallest Whisper model (whisper-tiny) is used, with key components initialized:
- Tokenizer: Converts text into tokens.
- Feature Extractor: Processes audio into features suitable for the model.
- Processor: Combines both tokenizer and feature extractor to handle both text and audio inputs.
2. Sample Data Processing:
- A sample from the dataset is loaded, including audio data and its transcription.
- The audio is processed to generate input features (encoded_input), while the transcription is tokenized into input IDs (encoded_label).
- The raw audio data, original transcription, processed features, and encoded labels are printed for verification.
3. Model Prediction Testing:
- The model generates a prediction using the processed audio features.
- The prediction is decoded and compared to the true transcription.
- Result: The predicted transcription ("Freeze my card please.") closely matches the actual label ("freeze my card please"), confirming that the processor works correctly.

#### Feature Extraction for the Full Dataset
1. Batch Processing Function:
- The prepare_datasets function is created to process audio and transcription for each batch.
- It extracts audio features, tokenizes text, and adds the audio length in seconds.
2. Processing the Entire Dataset:
- The merged_dataset is processed in batches, using parallel processing (4 processes) to speed up the operation.
- Unnecessary columns are removed after processing to keep only relevant features (input_features, labels, input_length).
3. Output Verification:
- The processed dataset, now with input_features, labels, and input_length, is printed to confirm the structure.
- Optional checks are done to ensure consistency in the shapes of input features and labels.

# Human Identification by the Iris

A Python implementation of an iris-based biometric recognition system using classical computer vision and machine learning (LDA). Built on top of the CASIA iris dataset.

> Based on: [smahesh29/OpenCV-Face-and-Eye-Detection](https://github.com/smahesh29/OpenCV-Face-and-Eye-Detection) and [akshatapatel/Iris-Recognition](https://github.com/akshatapatel/Iris-Recognition), modified for custom tasks.

## Pipeline

1. **Iris Localization** — detect and extract the iris region from eye images
2. **Iris Normalization** — unwrap iris to a fixed-size rectangular representation
3. **Image Enhancement** — improve contrast and quality for feature extraction
4. **Feature Extraction** — extract discriminative features using 1D Log-Gabor filters
5. **Iris Matching** — compare feature vectors using distance metrics
6. **Performance Evaluation** — compute recognition accuracy metrics

## Tech Stack

- Python 3.x
- OpenCV — image processing and face/eye detection
- NumPy, SciPy — numerical computations
- scikit-learn (LDA) — dimensionality reduction and matching
- Matplotlib — visualization
- Pandas — data handling

## Dataset

The project uses the [CASIA Iris Database](http://www.cbsr.ia.ac.cn/english/IrisDatabase.asp). Images should be placed in:

```
Eyes/
├── 001/
│   ├── 001_1_1.jpg   # training images (*_1_*)
│   ├── 001_2_1.jpg   # test images (*_2_*)
│   └── ...
├── 002/
└── ...
```

## Getting Started

### Installation

```bash
git clone https://github.com/vevdokimovm/Human-Identification-by-the-Iris.git
cd Human-Identification-by-the-Iris
pip install -r requirements.txt
```

### Run

```bash
python IrisRecognition.py
```

For real-time face/eye detection from camera:

```bash
python face_eye_detection_image.py
```

## Project Structure

```
Human-Identification-by-the-Iris/
├── IrisLocalization.py       # Iris boundary detection
├── IrisNormalization.py      # Daugman rubber sheet model
├── ImageEnhancement.py       # Histogram equalization
├── FeatureExtraction.py      # Log-Gabor feature extraction
├── IrisMatching.py           # Feature comparison
├── PerformanceEvaluation.py  # Accuracy metrics
├── IrisRecognition.py        # Main pipeline
├── face_eye_detection_image.py  # Real-time detection
├── haarcascade_*.xml         # Pre-trained Haar cascades
├── Eyes/                     # Training/test dataset
└── New_people/               # New subject images
```

## License

MIT

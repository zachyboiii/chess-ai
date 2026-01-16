# Chess AI

A deep learning-based chess engine trained on elite-level Lichess games. Play against the AI in a Streamlit web app, evaluate its performance against Stockfish, or dive into the training notebook to understand the approach.

## Quick Start

### Play Against the AI (Try it out!)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open your browser at `http://localhost:8501` and start playing!

## Project Overview

This project demonstrates a neural network approach to learning chess move prediction:

1. **Training Approach**: Train a CNN on historical chess games to predict strong moves
2. **Evaluation**: Use `predict.ipynb` to test the model against Stockfish engine
3. **Interactive Demo**: Play against the trained model in a Streamlit web app

## Training Approach

The model is trained using **supervised learning** on elite-level chess games:

### Data

- **Source**: Lichess elite player games (2013-2018+)
- **Format**: PGN (Portable Game Notation) files
- **Processing**: Each game is converted to board positions with corresponding strong moves

### Model Architecture

- **Input**: 13-channel 8×8 matrix (piece positions)
  - 6 channels for white pieces (pawn, knight, bishop, rook, queen, king)
  - 6 channels for black pieces
  - 1 channel for side to move
- **Architecture**: CNN → Fully Connected Layers
  - Conv2D (13→64, kernel=3) + ReLU
  - Conv2D (64→128, kernel=3) + ReLU
  - Flatten → Dense (8192→256) + ReLU → Dense (256→moves)

### Training

- **Framework**: PyTorch
- **Loss**: Cross-entropy (move classification)
- **Pre-trained Model**: `10Epoch_chessModel.pth` (10 epochs on Lichess data)
- **Device**: GPU-accelerated (CUDA) if available, CPU fallback

## Evaluation Against Stockfish

Use `predict.ipynb` to:

- Load the trained model and Stockfish engine
- Play test games between the Chess AI and Stockfish
- Analyze move quality and compare performance
- Evaluate win rates, accuracy, and move agreement

Run the notebook to see how the trained model performs against one of the strongest chess engines.

## Interactive App

The Streamlit app (`app.py`) provides a user-friendly interface to:

- **Play against the AI**: Make moves and let the model respond
- **Real-time board visualization**: See the current position and move history
- **Move tracking**: View all moves in the current game
- **GPU acceleration**: Automatic CUDA support for fast inference

Simply run:

```bash
streamlit run app.py
```

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
chess_ai/
├── app.py                    # Streamlit web app
├── model.py                  # CNN model definition
├── dataset.py               # Data loading & preprocessing
├── aux_func.py              # Board encoding utilities
├── train.ipynb              # Training notebook
├── predict.ipynb            # Stockfish evaluation notebook
├── requirements.txt         # Dependencies
├── models/
│   └── 10Epoch_chessModel.pth  # Pre-trained weights
├── data/
│   └── pgn/                 # Lichess elite games
└── stockfish/               # Stockfish chess engine
```

## Dependencies

- `chess`: Chess board and move validation
- `numpy`: Numerical computing
- `torch`: Deep learning framework
- `streamlit`: Web interface
- `tqdm`: Progress bars

See [requirements.txt](requirements.txt) for exact versions.

## Next Steps

- **Deploy**: The app is ready for deployment!
- **Improve**: Train for more epochs or with additional data
- **Optimize**: Experiment with different architectures
- **Benchmark**: Compare against other engines with `predict.ipynb`

## License

Please respect Lichess.org's data usage policies when using the training data.

---

**Ready to play?** Run `streamlit run app.py` and challenge the AI!

# Chess AI

A deep learning-based chess engine trained on elite-level Lichess games. Play against the AI in a Streamlit web app, evaluate its performance against Stockfish, or dive into the training notebook to understand the approach.

## Quick Start

### Play Against the AI (Try it out!)

Try it out [here](https://zac-chess-ai.streamlit.app/)!
`https://zac-chess-ai.streamlit.app/`

---

## Project Overview

This project demonstrates a neural network approach to learning chess move prediction:

1. **Training Approach**: Train a CNN on historical chess games to predict strong moves
2. **Evaluation**: Use `predict.ipynb` to test the model against Stockfish engine
3. **Interactive Demo**: Play against the trained model in a Streamlit web app

---

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

---

## Evaluation Against Stockfish

Use `predict.ipynb` to:

- Load the trained model and Stockfish engine
- Play test games between the Chess AI and Stockfish
- Analyze move quality and compare performance
- Evaluate win rates, accuracy, and move agreement

Run the notebook to see how the trained model performs against one of the strongest chess engines.

---

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
---

## License

Please respect Lichess.org's data usage policies when using the training data.


import streamlit as st
import torch
import pickle
from chess import Board, Move
import random
import chess.svg
from io import StringIO
from aux_func import board_to_matrix
from model import ChessModel

st.set_page_config(page_title="Chess AI Game", layout="wide")

# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = Board()
    st.session_state.move_history = []
    st.session_state.game_over = False
    st.session_state.result_message = ""
    st.session_state.player_is_white = True
    st.session_state.last_processed_move = None
    st.session_state.input_key = 0

# Load model and move dictionary
@st.cache_resource
def load_model_and_moves():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open('./models/move_to_int2.pkl', 'rb') as f:
        move_to_int = pickle.load(f)
    
    int_to_move = {v: k for k, v in move_to_int.items()}
    num_classes = len(move_to_int)
    
    model = ChessModel(num_classes).to(device)
    model.load_state_dict(torch.load('./models/50Epoch_chessModel.pth', map_location=device))
    model.eval()
    
    return model, move_to_int, int_to_move, device

model, move_to_int, int_to_move, device = load_model_and_moves()

def get_model_move(board):
    """Get the model's best move for the current board position."""
    with torch.no_grad():
        board_matrix = board_to_matrix(board)
        board_tensor = torch.tensor(board_matrix, dtype=torch.float32).unsqueeze(0).to(device)
        
        output = model(board_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        
        best_move = None
        best_score = -1
        
        for move_uci in legal_moves_uci:
            if move_uci in move_to_int:
                move_idx = move_to_int[move_uci]
                score = probabilities[move_idx].item()
                if score > best_score:
                    best_score = score
                    best_move = Move.from_uci(move_uci)
        
        if best_move is None:
            best_move = random.choice(legal_moves)
            best_score = 0.0
        
        return best_move, best_score

def display_board():
    """Display the chess board as larger SVG."""
    board = st.session_state.board
    board_svg = chess.svg.board(board, size=500)
    st.image(board_svg)

st.title("♟️ Chess AI - Play Against the Model")

# Sidebar controls
with st.sidebar:
    st.header("Game Controls")
    
    # Color selection (only at start)
    if not st.session_state.move_history:
        player_color = st.radio("Play as:", ["White", "Black"], horizontal=True)
        st.session_state.player_is_white = player_color == "White"
    else:
        st.write(f"Playing as: {'White' if st.session_state.player_is_white else 'Black'}")
    
    if st.button("New Game", use_container_width=True):
        st.session_state.board = Board()
        st.session_state.move_history = []
        st.session_state.game_over = False
        st.session_state.result_message = ""
        st.session_state.last_processed_move = None  # Reset for new game
        st.session_state.input_key += 1  # Clear input box
        st.rerun()
    
    if st.button("Undo Last Move", use_container_width=True):
        if len(st.session_state.move_history) >= 2:
            st.session_state.board.pop()
            st.session_state.board.pop()
            st.session_state.move_history = st.session_state.move_history[:-2]
            st.session_state.game_over = False
            st.session_state.last_processed_move = None  # Reset so moves can be repeated
            st.session_state.input_key += 1  # Clear input box
            st.rerun()
    
    st.divider()
    st.subheader("Move History")
    if st.session_state.move_history:
        moves_text = " ".join(st.session_state.move_history)
        st.text_area("Moves:", value=moves_text, height=150, disabled=True)
    else:
        st.text("No moves yet")

# Main game area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Board")
    display_board()

with col2:
    st.subheader("Game Status")
    
    if st.session_state.game_over:
        st.info(st.session_state.result_message)
    else:
        board = st.session_state.board
        is_player_turn = board.turn == st.session_state.player_is_white
        
        if board.is_check():
            st.warning("⚠️ Check!")
        
        if is_player_turn:
            # Input for player move
            player_move_input = st.text_input(
                "Enter your move:",
                key=f"move_input_{st.session_state.input_key}",
                placeholder="eg. e2e4, g1h3, etc."
            ).strip().lower()

            # Display legal moves
            legal_moves = list(board.legal_moves)
            if legal_moves:
                st.write(f"**Legal moves: {len(legal_moves)}**")
                moves_str = ", ".join([move.uci() for move in legal_moves])
                
                st.caption(moves_str)
            
            if player_move_input and player_move_input != st.session_state.last_processed_move:
                st.session_state.last_processed_move = player_move_input
                try:
                    # Try to parse the move
                    player_move = Move.from_uci(player_move_input)
                    
                    if player_move in legal_moves:
                        # Player makes move
                        st.session_state.board.push(player_move)
                        st.session_state.move_history.append(player_move_input)
                        
                        # Check if game is over
                        if st.session_state.board.is_game_over():
                            st.session_state.game_over = True
                            if st.session_state.board.is_checkmate():
                                st.session_state.result_message = "🎉 You won! Checkmate!"
                            elif st.session_state.board.is_stalemate():
                                st.session_state.result_message = "🤝 Draw! Stalemate."
                            else:
                                st.session_state.result_message = f"Game over: {st.session_state.board.outcome()}"
                        else:
                            
                            # Model makes move
                            model_move, confidence = get_model_move(st.session_state.board)
                            st.session_state.board.push(model_move)
                            st.session_state.move_history.append(model_move.uci())
                            st.session_state.input_key += 1  # Clear input after AI moves
                            st.session_state.last_processed_move = None  # Reset for next player turn
                            
                            st.info(f"🤖 Model played: **{model_move.uci()}** (confidence: {confidence:.1%})")
                            
                            # Check if game is over
                            if st.session_state.board.is_game_over():
                                st.session_state.game_over = True
                                if st.session_state.board.is_checkmate():
                                    st.session_state.result_message = "😢 Model won! Checkmate!"
                                elif st.session_state.board.is_stalemate():
                                    st.session_state.result_message = "🤝 Draw! Stalemate."
                                else:
                                    st.session_state.result_message = f"Game over: {st.session_state.board.outcome()}"

                        st.rerun()
                    else:
                        st.error(f"❌ Illegal move! {player_move_input} is not a legal move.")
                except Exception as e:
                    st.error(f"❌ Invalid move format. Use UCI notation (e.g., e2e4). Error: {str(e)}")
        else:
            st.write("**Waiting for model to move...**")
            # Model makes move
            model_move, confidence = get_model_move(board)
            st.session_state.board.push(model_move)
            st.session_state.move_history.append(model_move.uci())
            st.session_state.input_key += 1  # Clear input after AI moves
            st.session_state.last_processed_move = None  # Reset for next player turn
            
            st.info(f"🤖 Model played: **{model_move.uci()}** (confidence: {confidence:.1%})")
            
            # Check if game is over
            if st.session_state.board.is_game_over():
                st.session_state.game_over = True
                if st.session_state.board.is_checkmate():
                    st.session_state.result_message = "😢 Model won! Checkmate!"
                elif st.session_state.board.is_stalemate():
                    st.session_state.result_message = "🤝 Draw! Stalemate."
                else:
                    st.session_state.result_message = f"Game over: {st.session_state.board.outcome()}"
            
            st.rerun()

st.divider()
st.caption("♟️ Play chess against the trained AI model. Enter moves in UCI notation (e.g., e2e4).")

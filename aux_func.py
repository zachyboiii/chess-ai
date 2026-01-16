import numpy as np
from chess import Board
from chess import pgn

def board_to_matrix(board: Board):
    '''
    Change board to matrix form
    Args: board (Board type from chess lib)
    '''
    
    # 8x8 is size of chess board
    # 12 -> no. of unique pieces
    # 13 -> 12 pieces + 1 for legal moves
    
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()


    # Populate first 12 8x8 boards where the pieces are
    # '1' represents where the piece is
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1 # Each board has one unique piece

    # Populate legal moves board (13th board)
    # 1 represents where the piece can move
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square,8)
        matrix[12, row_to, col_to] = 1
    
    return matrix


def create_input_nn(games: pgn.Game):
    '''
    Create input for Neural Networks later
    For each game get initial board position
    
    '''
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci()) # UCI format: "e2e4" meaning move from e2 to e4
            board.push(move)
    return X, y

def encode_moves(moves):
    '''
    Convert moves from UCI format to numbers
    Returns:
    1. List of moves after applying mapping
    2. The mapping itself
    '''
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return [move_to_int[move] for move in moves], move_to_int
import chess
import numpy as np
import time
import random
# lucha hups
class Player:
    __slots__ = ['playerOne', 'depth', 'time_limit',
                 'piece_square_tables', 'piece_values']

    def __init__(self, playerOne: bool):
        """
        Initialize the player.
        """
        self.playerOne = playerOne  
        self.depth = 3  
        self.time_limit = 2  

        self.piece_square_tables = {
            chess.PAWN: np.array([
                0, 0, 0, 0, 0, 0, 0, 0,
                5, 10, 10, -20, -20, 10, 10, 5,
                5, -5, -10, 0, 0, -10, -5, 5,
                0, 0, 0, 20, 20, 0, 0, 0,
                5, 5, 10, 25, 25, 10, 5, 5,
                10, 10, 20, 30, 30, 20, 10, 10,
                50, 50, 50, 50, 50, 50, 50, 50,
                0, 0, 0, 0, 0, 0, 0, 0,
            ]),
            chess.KNIGHT: np.array([
                -50, -40, -30, -30, -30, -30, -40, -50,
                -40, -20, 0, 5, 5, 0, -20, -40,
                -30, 5, 10, 15, 15, 10, 5, -30,
                -30, 0, 15, 20, 20, 15, 0, -30,
                -30, 5, 15, 20, 20, 15, 5, -30,
                -30, 0, 10, 15, 15, 10, 0, -30,
                -40, -20, 0, 0, 0, 0, -20, -40,
                -50, -40, -30, -30, -30, -30, -40, -50,
            ]),
            chess.BISHOP: np.array([
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10, 5, 0, 0, 0, 0, 5, -10,
                -10, 10, 10, 10, 10, 10, 10, -10,
                -10, 0, 10, 10, 10, 10, 0, -10,
                -10, 5, 5, 10, 10, 5, 5, -10,
                -10, 0, 5, 10, 10, 5, 0, -10,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -20, -10, -10, -10, -10, -10, -10, -20,
            ]),
            chess.ROOK: np.array([
                0, 0, 0, 5, 5, 0, 0, 0,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                5, 10, 10, 10, 10, 10, 10, 5,
                0, 0, 0, 0, 0, 0, 0, 0,
            ]),
            chess.QUEEN: np.array([
                -20, -10, -10, -5, -5, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 5, 5, 5, 0, -10,
                -5, 0, 5, 5, 5, 5, 0, -5,
                0, 0, 5, 5, 5, 5, 0, -5,
                -10, 5, 5, 5, 5, 5, 0, -10,
                -10, 0, 5, 0, 0, 0, 0, -10,
                -20, -10, -10, -5, -5, -10, -10, -20,
            ]),
            chess.KING: np.array([
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -20, -30, -30, -40, -40, -30, -30, -20,
                -10, -20, -20, -20, -20, -20, -20, -10,
                20, 20, 0, 0, 0, 0, 20, 20,
                20, 30, 10, 0, 0, 10, 30, 20,
            ]),
        }

        #  values (indexed by piece type for faster access)
        self.piece_values = [0, 100, 320, 330, 500, 900, 20000]  

    def evaluate_board(self, board: chess.Board) -> int:
        """
        Evaluate the board state based on material and piece-square positioning.
        """
        if board.is_checkmate():
            return -999999 if board.turn else 999999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        material_score = 0
        positional_score = 0
        piece_map = board.piece_map()

        for square, piece in piece_map.items():
            piece_type = piece.piece_type
            color = piece.color

            # Use precomputed values
            piece_value = self.piece_values[piece_type]
            table = self.piece_square_tables[piece_type]

            if color == chess.WHITE:
                material_score += piece_value
                positional_score += table[square]
            else:
                material_score -= piece_value
                # Mirror the square for black pieces
                positional_score -= table[chess.square_mirror(square)]

        return material_score + positional_score

    def order_moves(self, board):
        """
        Order moves to improve Alpha-Beta pruning using MVV-LVA heuristic.
        """
        moves = list(board.legal_moves)
        piece_values = self.piece_values

        def move_order(move):
            score = 0
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                if captured_piece and attacker_piece:
                    # MVV-LVA: Most Valuable Victim - Least Valuable Aggressor
                    score += 10 * piece_values[captured_piece.piece_type] - piece_values[attacker_piece.piece_type]
            if board.gives_check(move):
                score += 1000  # moves that give check
            if move.promotion:
                score += 5000  #  promotions
            return score

        moves.sort(key=move_order, reverse=True)
        return moves

    def pvs(self, board, depth, alpha, beta, color):
        """
        Principal Variation Search algorithm with alpha-beta pruning.
        """
        if depth == 0 or board.is_game_over():
            return color * self.evaluate_board(board)

        b_search_pv = True
        for move in self.order_moves(board):
            board.push(move)
            if b_search_pv:
                score = -self.pvs(board, depth - 1, -beta, -alpha, -color)
            else:
                #  window search
                score = -self.pvs(board, depth - 1, -alpha - 1, -alpha, -color)
                if alpha < score < beta:
                    # If it failed high, do a full re-search
                    score = -self.pvs(board, depth - 1, -beta, -alpha, -color)
            board.pop()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score
                b_search_pv = False  # Found a better move

        return alpha

    def makeMove(self, gameState: chess.Board) -> chess.Move:
        """
        Determine the best move using PVS with iterative deepening and aspiration windows.
        """
        best_move = None
        start_time = time.time()
        board = gameState.copy()
        color = 1 if self.playerOne else -1

        best_value = 0  # Initial best value for aspiration window
        delta = 50  # Initial aspiration window size

        for depth in range(1, self.depth + 1):
            if time.time() - start_time >= self.time_limit - 0.1:
                break  # Time limit reached

            alpha = best_value - delta
            beta = best_value + delta

            while True:
                current_best_move = None
                current_best_value = -np.inf

                for move in self.order_moves(board):
                    if time.time() - start_time >= self.time_limit - 0.1:
                        break  # Time limit reached

                    board.push(move)
                    score = -self.pvs(board, depth - 1, -beta, -alpha, -color)
                    board.pop()

                    if score > current_best_value:
                        current_best_value = score
                        current_best_move = move

                    if score > alpha:
                        alpha = score
                    if alpha >= beta:
                        break  # Beta cutoff

                if current_best_move:
                    best_move = current_best_move
                    best_value = current_best_value

                # Aspiration window adjustment
                if alpha >= beta:
                    #increase window and re-search
                    delta *= 2
                    alpha = -np.inf
                    beta = np.inf
                    continue
                elif alpha <= best_value - delta:
                    # increase window and re-search
                    delta *= 2
                    alpha = -np.inf
                    beta = np.inf
                    continue
                else:
                    # Successful search
                    break

        # dont judge haha
        return best_move or random.choice(list(gameState.legal_moves))


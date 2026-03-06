from chess_tournament import Player
import chess
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn.functional as F


class TransformerPlayer(Player):

    def __init__(self, name="MyPlayer"):
        super().__init__(name)

        model_name = "Mariyana0019/chess-tinyllama-combined"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        print(f"✅ {self.name} is ready!")

    def score_move(self, prompt, move):
        full_text = prompt + " " + move
        inputs = self.tokenizer(full_text, return_tensors="pt").to("cuda")
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = prompt_inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits[0, prompt_length-1:-1]
            labels = inputs["input_ids"][0, prompt_length:]
            loss = F.cross_entropy(logits, labels)
        return -loss.item()

    def piece_value(self, board, move):
        piece = board.piece_at(move.to_square)
        if piece is None:
            return 0
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100
        }
        return values.get(piece.piece_type, 0)

    def get_move(self, fen):
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        # Priority 1: Checkmate — take it immediately!
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        # Priority 2: Best capture (take the most valuable piece)
        captures = [(move, self.piece_value(board, move))
                    for move in legal_moves if board.is_capture(move)]
        if captures:
            best_capture = max(captures, key=lambda x: x[1])
            if best_capture[1] >= 3:  # only take if worth at least a knight
                return best_capture[0].uci()

        # Priority 3: Give check
        for move in legal_moves:
            board.push(move)
            if board.is_check():
                board.pop()
                return move.uci()
            board.pop()

        # Priority 4: Use model score for everything else
        prompt = f"[INST] Chess move for: {fen} [/INST]"
        best_move = None
        best_score = float('-inf')
        for move in legal_moves:
            score = self.score_move(prompt, move.uci())
            if score > best_score:
                best_score = score
                best_move = move.uci()

        return best_move

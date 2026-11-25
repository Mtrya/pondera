"""Architecture-independent autoregressive generation and FINAL answer parsing."""

import time
from dataclasses import dataclass, field

import chess
import torch

from pondera.chess.codec import decode_value_token, encode_board
from pondera.chess.merge import MergeDictionary, MergeDictionaryError, decode
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    CHESS_VOCAB_SIZE,
    END,
    FINAL,
    IDX_TO_UCI_MOVE,
    MATE_LOSS,
    MATE_WIN,
    MOVE,
    PAD,
    VALUE,
    VALUE_BINS,
)
from pondera.models import PonderaModel


@dataclass
class Answer:
    final_found: bool = False
    end_found: bool = False
    format_valid: bool = False
    move_legal: bool = False
    pv_legal: bool = False
    decision: str | None = None
    value_bin: int | None = None
    value: float | None = None
    mate_distance: int | None = None
    pv: list[str] = field(default_factory=list)
    legal_pv_plies: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self):
        return self.format_valid and self.move_legal and self.pv_legal


def parse_answer(tokens, dictionary, fen, repetition=1):
    """Parse expanded atoms; the reasoning before FINAL is deliberately opaque.

    No canonical tokenization is required. A composite may cross value/PV or
    decision/value boundaries. PV moves follow the decision, in the root frame.
    """
    answer = Answer()
    try:
        atoms = decode(tokens, dictionary).tolist()
    except (MergeDictionaryError, IndexError):
        answer.errors.append("invalid_token_id")
        return answer
    finals = [index for index, atom in enumerate(atoms) if atom == FINAL]
    answer.final_found = bool(finals)
    answer.end_found = END in atoms
    if not finals:
        answer.errors.append("missing_final")
        return answer
    if len(finals) != 1:
        answer.errors.append("multiple_final")
        return answer
    suffix = atoms[finals[0] + 1 :]
    end_after_final = END in suffix
    if not end_after_final:
        answer.errors.append("missing_end")
    elif suffix[-1] != END or suffix.count(END) != 1:
        answer.errors.append("trailing_after_end")
    body = suffix[: suffix.index(END)] if end_after_final else suffix
    if not body or not MOVE <= body[0] < VALUE:
        answer.errors.append("missing_decision")
        return answer
    board = chess.Board(fen)
    mirrored = board.turn == chess.BLACK

    def move_name(atom):
        move = IDX_TO_UCI_MOVE[atom - MOVE]
        return reflect_uci(move) if mirrored and move != "<claim_draw>" else move

    answer.decision = move_name(body[0])
    claim = answer.decision == "<claim_draw>"
    answer.move_legal = (
        not board.is_game_over()
        and repetition < 5
        and (
            repetition >= 3 or board.halfmove_clock >= 100
            if claim
            else chess.Move.from_uci(answer.decision) in board.legal_moves
        )
    )
    if not answer.move_legal:
        answer.errors.append("illegal_decision")
    if len(body) < 2 or not VALUE <= body[1] < MATE_WIN:
        answer.errors.append("missing_value")
        return answer
    answer.value_bin = body[1] - VALUE
    answer.value = decode_value_token(body[1])
    position = 2
    if position < len(body) and MATE_WIN <= body[position] < PAD:
        atom = body[position]
        win = atom < MATE_LOSS
        answer.mate_distance = atom - MATE_WIN if win else -(atom - MATE_LOSS)
        if answer.value_bin != (VALUE_BINS - 1 if win else 0):
            answer.errors.append("mate_value_mismatch")
        position += 1
    for atom in body[position:]:
        if not MOVE <= atom < VALUE or IDX_TO_UCI_MOVE[atom - MOVE] == "<claim_draw>":
            answer.errors.append("invalid_pv_token")
            return answer
        answer.pv.append(move_name(atom))
    answer.format_valid = not any(
        error != "illegal_decision" for error in answer.errors
    )
    if claim:
        answer.pv_legal = answer.move_legal and not answer.pv
        if answer.pv:
            answer.format_valid = False
            answer.errors.append("claim_with_pv")
        return answer
    if not answer.move_legal:
        return answer
    board.push_uci(answer.decision)
    for move in answer.pv:
        parsed = chess.Move.from_uci(move)
        if board.is_game_over() or parsed not in board.legal_moves:
            answer.errors.append("illegal_pv")
            return answer
        board.push(parsed)
        answer.legal_pv_plies += 1
    answer.pv_legal = True
    return answer


@dataclass
class Generation:
    tokens: list[int]
    answer: Answer
    reasoning_tokens: int
    answer_tokens: int
    generated_tokens: int
    forced_final: bool
    stop: str
    seconds: float
    expanded_atoms: int


@dataclass
class _DecodeState:
    caches: tuple | None = None
    consumed: int = 0


@dataclass
class Predictor:
    model: torch.nn.Module
    dictionary: MergeDictionary
    metadata: dict

    def _next_logits(self, prefix, tokens, state):
        device = next(self.model.parameters()).device
        context = prefix + tokens
        # Cached attention supports one new token at a time. Inserting FINAL
        # can leave two unseen tokens; prefill that context together so neither
        # token attends to its future. A replaced reasoning prefix also resets.
        if len(context) != state.consumed + 1:
            state.caches = None
        unseen = context if state.caches is None else context[state.consumed :]
        ids = torch.tensor(unseen, dtype=torch.long, device=device)
        offsets = torch.tensor([0, len(context)], dtype=torch.int32)
        logits, state.caches = self.model.forward_cached(
            ids, offsets, state.caches, host_tokens=unseen
        )
        state.consumed = len(context)
        return logits[-1]

    def _step(self, prefix, tokens, state):
        token = int(self._next_logits(prefix, tokens, state).argmax())
        tokens.append(token)
        if token >= CHESS_VOCAB_SIZE:
            return self.dictionary.expansion(token)
        return (token,)

    def _reason(self, prefix, tokens, budget, state):
        for count in range(1, budget + 1):
            expanded = self._step(prefix, tokens, state)
            if FINAL in expanded:
                return count, False, "end" if END in expanded else "answer_budget"
            if END in expanded:
                return count, False, "end_before_final"
        tokens.append(FINAL)
        return budget, True, "answer_budget"

    @torch.inference_mode()
    def decision_logits(self, fen, repetition=1, *, reasoning_tokens=0):
        """Return the root-decision logits after generating a reasoning prefix."""
        if reasoning_tokens < 0:
            raise ValueError("Require nonnegative reasoning tokens")
        self.model.eval()
        prefix = encode_board(fen, repetition).tolist()
        tokens = []
        state = _DecodeState()
        _, _, stop = self._reason(prefix, tokens, reasoning_tokens, state)
        if stop != "answer_budget":
            tokens = [FINAL]
            state = _DecodeState()
        return self._next_logits(prefix, tokens, state)

    @torch.inference_mode()
    def generate(self, fen, repetition=1, *, reasoning_tokens=0, answer_tokens=128):
        """Greedy cached decoding, with no masking or answer repair.

        Budgets count model calls, not expanded atoms. The answer allowance
        counts tokens after FINAL. A model-emitted FINAL consumes a reasoning
        call; inserting FINAL at the limit consumes no call. END is never forced.
        """
        if reasoning_tokens < 0 or answer_tokens < 1:
            raise ValueError(
                "Require nonnegative reasoning and positive answer budgets"
            )
        prefix = encode_board(fen, repetition).tolist()
        tokens = []
        answer_count = 0
        self.model.eval()
        started = time.monotonic()
        state = _DecodeState()

        reasoning_count, forced, stop = self._reason(
            prefix, tokens, reasoning_tokens, state
        )
        if stop == "answer_budget":
            for _ in range(answer_tokens):
                expanded = self._step(prefix, tokens, state)
                answer_count += 1
                if END in expanded:
                    stop = "end"
                    break
        return Generation(
            tokens=tokens,
            answer=parse_answer(tokens, self.dictionary, fen, repetition),
            reasoning_tokens=reasoning_count,
            answer_tokens=answer_count,
            generated_tokens=reasoning_count + answer_count,
            forced_final=forced,
            stop=stop,
            seconds=time.monotonic() - started,
            expanded_atoms=len(decode(tokens, self.dictionary)),
        )


def load_checkpoint(path, device="cpu", *, loop_repetitions=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["arch"] != PonderaModel.architecture:
        raise ValueError(f"Unsupported checkpoint architecture: {state['arch']}")
    model = PonderaModel(**state["model_config"])
    model.load_state_dict(state["model_state_dict"])
    if loop_repetitions is not None:
        model.override_repetitions(loop_repetitions)
    identity = state.get("tokenizer_identity", {})
    dictionary = MergeDictionary(identity.get("merge_entries", ()))
    if model.vocab_size != CHESS_VOCAB_SIZE + len(dictionary):
        raise ValueError("Checkpoint vocabulary and merge entries disagree")
    model.bind_tokenizer(dictionary)
    metadata = {
        key: state[key]
        for key in ("arch", "model_config", "update", "presented_documents")
        if key in state
    }
    metadata["tokenizer_identity"] = identity
    if model.loop_segments:
        metadata["segment_repetitions"] = model.repetitions
    return Predictor(model.to(device).eval(), dictionary, metadata)

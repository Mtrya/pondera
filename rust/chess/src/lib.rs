mod board;
mod pgn;

use board::{Ply, Position, START};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pyo3::create_exception!(_rules, IllegalMoveError, PyValueError);

#[pyclass(frozen, eq, hash, skip_from_py_object, module = "pondera.chess._rules")]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Move(Ply);

#[pymethods]
impl Move {
    #[staticmethod]
    fn from_uci(text: &str) -> PyResult<Self> {
        Ply::parse(text)
            .map(Self)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid UCI move: {text}")))
    }

    fn uci(&self) -> String {
        self.0.uci()
    }

    fn __repr__(&self) -> String {
        format!("Move({:?})", self.uci())
    }
}

#[pyclass(skip_from_py_object, module = "pondera.chess._rules")]
#[derive(Clone)]
struct Board {
    position: Position,
    history: Vec<(Position, Ply)>,
}

impl Board {
    fn at(position: Position) -> Self {
        Self {
            position,
            history: Vec::new(),
        }
    }

    fn require_legal(&self, ply: Ply) -> PyResult<()> {
        if self.position.legal(ply) {
            Ok(())
        } else {
            Err(IllegalMoveError::new_err(format!(
                "Illegal move {} in {}",
                ply.uci(),
                self.position.fen(true)
            )))
        }
    }

    fn apply(&mut self, ply: Ply) {
        self.history.push((self.position, ply));
        self.position = self.position.after(ply);
    }

    fn occurrences(&self, limit: usize) -> usize {
        let key = self.position.key();
        let mut count = 1;
        for (previous, _) in self
            .history
            .iter()
            .rev()
            .take(self.position.halfmove as usize)
        {
            if count == limit {
                break;
            }
            if previous.white == self.position.white && previous.key() == key {
                count += 1;
            }
        }
        count
    }
}

#[pymethods]
impl Board {
    #[new]
    #[pyo3(signature = (fen=START))]
    fn new(fen: &str) -> PyResult<Self> {
        Position::from_fen(fen)
            .map(Self::at)
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn turn(&self) -> bool {
        self.position.white
    }

    #[getter]
    fn halfmove_clock(&self) -> u32 {
        self.position.halfmove
    }

    #[getter]
    fn fullmove_number(&self) -> u32 {
        self.position.fullmove
    }

    #[getter]
    fn legal_moves(&self) -> Vec<Move> {
        self.position.legal_moves().into_iter().map(Move).collect()
    }

    fn is_legal(&self, ply: &Move) -> bool {
        self.position.legal(ply.0)
    }

    fn has_legal_moves(&self) -> bool {
        self.position.has_moves()
    }

    fn legal_move_count(&self) -> usize {
        self.position.legal_move_count()
    }

    #[getter]
    fn move_stack(&self) -> Vec<Move> {
        self.history.iter().map(|(_, ply)| Move(*ply)).collect()
    }

    #[pyo3(signature = (*, stack=true))]
    fn copy(&self, stack: bool) -> Self {
        if stack {
            self.clone()
        } else {
            Self::at(self.position)
        }
    }

    fn root(&self) -> Self {
        Self::at(
            self.history
                .first()
                .map_or(self.position, |(root, _)| *root),
        )
    }

    #[pyo3(signature = (*, en_passant="legal"))]
    fn fen(&self, en_passant: &str) -> PyResult<String> {
        match en_passant {
            "fen" => Ok(self.position.fen(true)),
            "legal" => Ok(self.position.fen(false)),
            _ => Err(PyValueError::new_err("Choose fen or legal en passant")),
        }
    }

    fn board_fen(&self) -> String {
        self.position.placement()
    }

    fn position_key<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.position.key().bytes())
    }

    fn is_valid(&self) -> bool {
        self.position.is_valid()
    }

    fn is_check(&self) -> bool {
        self.position.is_check()
    }

    fn is_checkmate(&self) -> bool {
        self.is_check() && !self.has_legal_moves()
    }

    fn is_stalemate(&self) -> bool {
        !self.is_check() && !self.has_legal_moves()
    }

    fn is_insufficient_material(&self) -> bool {
        self.position.insufficient_material()
    }

    #[pyo3(signature = (count=3))]
    fn is_repetition(&self, count: usize) -> bool {
        self.occurrences(count) >= count
    }

    #[pyo3(signature = (limit=5))]
    fn repetition_count(&self, limit: usize) -> usize {
        self.occurrences(limit.max(1))
    }

    fn is_game_over(&self) -> bool {
        self.is_insufficient_material()
            || self.halfmove_clock() >= 150
            || !self.has_legal_moves()
            || self.is_repetition(5)
    }

    fn push(&mut self, ply: &Move) -> PyResult<()> {
        self.require_legal(ply.0)?;
        self.apply(ply.0);
        Ok(())
    }

    fn pop(&mut self) -> PyResult<Move> {
        let (position, ply) = self
            .history
            .pop()
            .ok_or_else(|| PyIndexError::new_err("No move to undo"))?;
        self.position = position;
        Ok(Move(ply))
    }

    fn push_uci(&mut self, text: &str) -> PyResult<Move> {
        let ply = Move::from_uci(text)?;
        self.push(&ply)?;
        Ok(ply)
    }

    fn san(&self, ply: &Move) -> PyResult<String> {
        self.require_legal(ply.0)?;
        Ok(self.position.san(ply.0))
    }

    fn san_and_push(&mut self, ply: &Move) -> PyResult<String> {
        let san = self.san(ply)?;
        self.apply(ply.0);
        Ok(san)
    }

    fn parse_san(&self, text: &str) -> PyResult<Move> {
        self.position
            .parse_san(text)
            .map(Move)
            .map_err(|kind| PyValueError::new_err(format!("{kind}: {text}")))
    }
}

#[pymodule]
fn _rules(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Board>()?;
    module.add_class::<Move>()?;
    module.add(
        "IllegalMoveError",
        module.py().get_type::<IllegalMoveError>(),
    )?;
    module.add("STARTING_FEN", START)?;
    module.add("WHITE", true)?;
    module.add("BLACK", false)?;
    module.add_function(wrap_pyfunction!(pgn::replay_mainline, module)?)?;
    Ok(())
}

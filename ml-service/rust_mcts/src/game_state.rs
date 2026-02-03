use pyo3::prelude::*;

/// Minimal game state for move encoding/decoding
#[pyclass]
#[derive(Clone)]
pub struct GameState {
    /// Current player (0-3)
    #[pyo3(get, set)]
    pub current_player: u8,

    /// Board state as flat array (20x20 = 400 cells)
    /// Values: 0=empty, 1-4=player pieces
    board: Vec<u8>,
}

#[pymethods]
impl GameState {
    #[new]
    pub fn new() -> Self {
        Self {
            current_player: 0,
            board: vec![0; 400],
        }
    }

    /// Create from Python board array
    #[staticmethod]
    pub fn from_board(board: Vec<u8>, current_player: u8) -> Self {
        Self {
            current_player,
            board,
        }
    }

    /// Encode a move as u32: piece_idx(8) | orientation(8) | row(8) | col(8)
    #[staticmethod]
    pub fn encode_move(piece_idx: u8, orientation: u8, row: u8, col: u8) -> u32 {
        ((piece_idx as u32) << 24)
            | ((orientation as u32) << 16)
            | ((row as u32) << 8)
            | (col as u32)
    }

    /// Decode a move from u32
    #[staticmethod]
    pub fn decode_move(move_id: u32) -> (u8, u8, u8, u8) {
        let piece_idx = ((move_id >> 24) & 0xFF) as u8;
        let orientation = ((move_id >> 16) & 0xFF) as u8;
        let row = ((move_id >> 8) & 0xFF) as u8;
        let col = (move_id & 0xFF) as u8;
        (piece_idx, orientation, row, col)
    }

    /// Get board as Python list
    pub fn get_board(&self) -> Vec<u8> {
        self.board.clone()
    }

    /// Set board from Python list
    pub fn set_board(&mut self, board: Vec<u8>) {
        self.board = board;
    }
}

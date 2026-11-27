//! Replay one standard-start PGN game, returning only the mainline positions.

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::board::{Position, START};

type RecordedPosition = (String, usize, u32, u32);
type Replay = (Option<&'static str>, Vec<RecordedPosition>, bool);

fn reject(reason: &'static str) -> Replay {
    (Some(reason), Vec::new(), false)
}

fn tag(line: &str) -> Option<(&str, &str)> {
    let line = line.strip_prefix('[')?.strip_suffix(']')?;
    let (name, value) = line.split_once(char::is_whitespace)?;
    Some((name, value.trim().strip_prefix('"')?.strip_suffix('"')?))
}

fn replay(game: &str) -> Replay {
    let mut initial_tags = true;
    let mut has_setup = false;
    let mut variant = false;
    let mut comment = false;
    let mut board = Position::from_fen(START).unwrap();
    let mut parent = None;
    let mut branches = Vec::new();
    let mut history = HashMap::from([(board.key(), 1)]);
    let mut rows = Vec::new();

    for line in game.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        if initial_tags && !comment {
            if let Some((name, value)) = tag(line) {
                variant |= name == "Variant" && !value.eq_ignore_ascii_case("Standard");
                has_setup |= matches!(name, "SetUp" | "FEN");
                continue;
            }
        }
        let mut tail = line;
        while !tail.is_empty() {
            if comment {
                let Some((_, rest)) = tail.split_once('}') else {
                    break;
                };
                tail = rest;
                comment = false;
                continue;
            }
            tail = tail.trim_start();
            let Some(&byte) = tail.as_bytes().first() else {
                break;
            };
            match byte {
                b';' => break,
                b'{' => {
                    comment = true;
                    tail = &tail[1..];
                }
                b'(' => {
                    let Some(before) = parent else {
                        return reject("illegal");
                    };
                    branches.push((board, parent));
                    board = before;
                    parent = None;
                    tail = &tail[1..];
                }
                b')' => {
                    let Some((main, previous)) = branches.pop() else {
                        return reject("illegal");
                    };
                    board = main;
                    parent = previous;
                    tail = &tail[1..];
                }
                b'$' => {
                    tail = tail[1..].trim_start_matches(|ch: char| ch.is_ascii_digit());
                }
                _ => {
                    let end = tail
                        .find(|ch: char| ch.is_whitespace() || "{}();$".contains(ch))
                        .unwrap_or(tail.len());
                    let word = &tail[..end];
                    tail = &tail[end..];
                    let movetext = word.trim_start_matches(|ch: char| ch.is_ascii_digit());
                    let word = if movetext.starts_with('.') {
                        movetext.trim_start_matches('.')
                    } else {
                        word
                    };
                    if word.is_empty() || matches!(word, "*" | "1-0" | "0-1" | "1/2-1/2" | "e.p.") {
                        continue;
                    }
                    if initial_tags {
                        if variant {
                            return reject("variant");
                        }
                        if has_setup {
                            return reject("setup");
                        }
                        initial_tags = false;
                    }
                    let Ok(ply) = board.parse_san(word) else {
                        return reject("illegal");
                    };
                    parent = Some(board);
                    board = board.after(ply);
                    if branches.is_empty() {
                        let count = history.entry(board.key()).or_default();
                        *count += 1;
                        rows.push((
                            board.position_fen(false),
                            *count,
                            board.halfmove,
                            board.fullmove,
                        ));
                    }
                }
            }
        }
    }
    if variant {
        return reject("variant");
    }
    if has_setup {
        return reject("setup");
    }
    if !branches.is_empty() {
        return reject("illegal");
    }
    let terminal = !rows.is_empty() && !board.has_moves();
    (None, rows, terminal)
}

#[pyfunction]
pub fn replay_mainline(py: Python<'_>, game: &str) -> Replay {
    py.detach(|| replay(game))
}

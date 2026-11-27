//! Standard chess position, move generation, and notation.

use std::fmt::Write;

pub const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const PAWN: i8 = 1;
const KNIGHT: i8 = 2;
const BISHOP: i8 = 3;
const ROOK: i8 = 4;
const QUEEN: i8 = 5;
const KING: i8 = 6;
const PIECES: &[u8; 7] = b" PNBRQK";
const ORTHOGONAL: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAGONAL: [(i8, i8); 4] = [(1, 1), (-1, 1), (1, -1), (-1, -1)];
const JUMPS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (-1, 2),
    (-2, 1),
    (1, -2),
    (2, -1),
    (-1, -2),
    (-2, -1),
];

pub fn square(bytes: &[u8]) -> Option<u8> {
    match bytes {
        [file @ b'a'..=b'h', rank @ b'1'..=b'8'] => Some((rank - b'1') * 8 + file - b'a'),
        _ => None,
    }
}

fn square_name(sq: u8) -> [char; 2] {
    [(b'a' + sq % 8) as char, (b'1' + sq / 8) as char]
}

fn offset(sq: u8, dx: i8, dy: i8) -> Option<u8> {
    let x = (sq % 8) as i8 + dx;
    let y = (sq / 8) as i8 + dy;
    ((0..8).contains(&x) && (0..8).contains(&y)).then_some((y * 8 + x) as u8)
}

pub fn symbol(piece: i8) -> char {
    let name = PIECES[piece.unsigned_abs() as usize];
    (if piece < 0 {
        name.to_ascii_lowercase()
    } else {
        name
    }) as char
}

fn piece_index(name: u8) -> Option<i8> {
    let kind = PIECES[1..]
        .iter()
        .position(|&p| p == name.to_ascii_uppercase())? as i8
        + 1;
    Some(if name.is_ascii_lowercase() {
        -kind
    } else {
        kind
    })
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Ply {
    pub from: u8,
    pub to: u8,
    pub promotion: i8,
}

impl Ply {
    pub fn parse(text: &str) -> Option<Self> {
        let bytes = text.as_bytes();
        if !matches!(bytes.len(), 4 | 5) {
            return None;
        }
        let from = square(&bytes[..2])?;
        let to = square(&bytes[2..4])?;
        if from == to {
            return None;
        }
        let promotion = match bytes.get(4) {
            None => 0,
            Some(b'n') => KNIGHT,
            Some(b'b') => BISHOP,
            Some(b'r') => ROOK,
            Some(b'q') => QUEEN,
            _ => return None,
        };
        Some(Self {
            from,
            to,
            promotion,
        })
    }

    pub fn uci(self) -> String {
        let mut name = String::with_capacity(5);
        name.extend(square_name(self.from));
        name.extend(square_name(self.to));
        if self.promotion != 0 {
            name.push(symbol(-self.promotion));
        }
        name
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct PositionKey {
    pieces: [i8; 64],
    white: bool,
    rights: u8,
    ep: Option<u8>,
}

impl PositionKey {
    pub fn bytes(self) -> [u8; 67] {
        let mut key = [0; 67];
        for (output, piece) in key.iter_mut().zip(self.pieces) {
            *output = (piece + KING) as u8;
        }
        key[64] = self.white.into();
        key[65] = self.rights;
        key[66] = self.ep.unwrap_or(64);
        key
    }
}

#[derive(Clone, Copy)]
pub struct Position {
    pub pieces: [i8; 64],
    pub white: bool,
    rights: u8,
    ep: Option<u8>,
    pub halfmove: u32,
    pub fullmove: u32,
}

impl Position {
    pub fn from_fen(fen: &str) -> Result<Self, &'static str> {
        let mut fields = fen.split_ascii_whitespace();
        let placement = fields.next().ok_or("FEN needs piece placement")?;
        let mut pieces = [0; 64];
        let mut ranks = placement.split('/').rev();
        for rank in pieces.chunks_exact_mut(8) {
            let mut file = 0;
            let mut gap = false;
            for byte in ranks.next().ok_or("FEN needs eight ranks")?.bytes() {
                match byte {
                    b'1'..=b'8' if !gap => {
                        file += (byte - b'0') as usize;
                        gap = true;
                    }
                    _ if file < 8 => {
                        rank[file] = piece_index(byte).ok_or("Invalid FEN placement")?;
                        file += 1;
                        gap = false;
                    }
                    _ => return Err("Invalid FEN placement"),
                }
            }
            if file != 8 {
                return Err("Each FEN rank needs eight squares");
            }
        }
        if ranks.next().is_some() {
            return Err("FEN needs eight ranks");
        }
        let white = match fields.next().unwrap_or("w") {
            "w" => true,
            "b" => false,
            _ => return Err("Invalid FEN turn"),
        };
        let mut rights = 0;
        let castling = fields.next().unwrap_or("-");
        if castling != "-" {
            for letter in castling.bytes() {
                let index = b"KQkq"
                    .iter()
                    .position(|&x| x == letter)
                    .ok_or("Invalid FEN castling rights")?;
                if rights & (1 << index) != 0 {
                    return Err("Duplicate FEN castling right");
                }
                rights |= 1 << index;
            }
        }
        let ep = match fields.next().unwrap_or("-") {
            "-" => None,
            text => Some(square(text.as_bytes()).ok_or("Invalid FEN en passant square")?),
        };
        let halfmove = fields
            .next()
            .unwrap_or("0")
            .parse()
            .map_err(|_| "Invalid FEN halfmove clock")?;
        let fullmove = fields
            .next()
            .unwrap_or("1")
            .parse::<u32>()
            .map_err(|_| "Invalid FEN fullmove number")?
            .max(1);
        if fields.next().is_some() {
            return Err("Unexpected FEN field");
        }
        Ok(Self {
            pieces,
            white,
            rights,
            ep,
            halfmove,
            fullmove,
        })
    }

    fn color(self) -> i8 {
        if self.white {
            1
        } else {
            -1
        }
    }

    fn at(&self, sq: u8) -> i8 {
        self.pieces[sq as usize]
    }

    pub fn placement(&self) -> String {
        let mut out = String::with_capacity(72);
        for (index, rank) in self.pieces.chunks_exact(8).rev().enumerate() {
            if index > 0 {
                out.push('/');
            }
            let mut empty = 0;
            for &piece in rank {
                if piece == 0 {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    out.push((b'0' + empty) as char);
                    empty = 0;
                }
                out.push(symbol(piece));
            }
            if empty > 0 {
                out.push((b'0' + empty) as char);
            }
        }
        out
    }

    fn clean_rights(&self) -> u8 {
        let mut rights = self.rights;
        for (home, color, shift) in [(0, 1, 0), (56, -1, 2)] {
            if self.at(home + 4) != color * KING {
                rights &= !(3 << shift);
            }
            if self.at(home + 7) != color * ROOK {
                rights &= !(1 << shift);
            }
            if self.at(home) != color * ROOK {
                rights &= !(2 << shift);
            }
        }
        rights
    }

    pub fn position_fen(&self, raw_ep: bool) -> String {
        let mut out = self.placement();
        out.push_str(if self.white { " w " } else { " b " });
        let rights = self.clean_rights();
        if rights == 0 {
            out.push('-');
        } else {
            for (index, letter) in "KQkq".chars().enumerate() {
                if rights & (1 << index) != 0 {
                    out.push(letter);
                }
            }
        }
        out.push(' ');
        if let Some(sq) = if raw_ep { self.ep } else { self.legal_ep() } {
            out.extend(square_name(sq));
        } else {
            out.push('-');
        }
        out
    }

    pub fn fen(&self, raw_ep: bool) -> String {
        let mut out = self.position_fen(raw_ep);
        write!(out, " {} {}", self.halfmove, self.fullmove).unwrap();
        out
    }

    fn king(&self, color: i8) -> Option<u8> {
        self.pieces
            .iter()
            .position(|&p| p == color * KING)
            .map(|sq| sq as u8)
    }

    fn attackers(&self, sq: u8, by: i8) -> u64 {
        let mut attackers = 0;
        for dx in [-1, 1] {
            if let Some(from) = offset(sq, dx, -by).filter(|&from| self.at(from) == by * PAWN) {
                attackers |= 1 << from;
            }
        }
        for (dx, dy) in JUMPS {
            if let Some(from) = offset(sq, dx, dy).filter(|&from| self.at(from) == by * KNIGHT) {
                attackers |= 1 << from;
            }
        }
        for (directions, slider) in [(ORTHOGONAL, ROOK), (DIAGONAL, BISHOP)] {
            for (dx, dy) in directions {
                let mut cursor = sq;
                while let Some(from) = offset(cursor, dx, dy) {
                    let piece = self.at(from) * by;
                    if piece != 0 {
                        if piece == slider || piece == QUEEN || (piece == KING && cursor == sq) {
                            attackers |= 1 << from;
                        }
                        break;
                    }
                    cursor = from;
                }
            }
        }
        attackers
    }

    pub fn in_check(&self, color: i8) -> bool {
        self.king(color)
            .is_some_and(|sq| self.attackers(sq, -color) != 0)
    }

    pub fn is_check(&self) -> bool {
        self.in_check(self.color())
    }

    fn valid_ep(&self) -> Option<u8> {
        let ep = self.ep?;
        let side = self.color();
        let victim = offset(ep, 0, -side)?;
        let origin = offset(ep, 0, side)?;
        (ep / 8 == if self.white { 5 } else { 2 })
            .then_some(ep)
            .filter(|&sq| {
                self.at(sq) == 0 && self.at(origin) == 0 && self.at(victim) == -side * PAWN
            })
    }

    pub fn is_valid(&self) -> bool {
        if self.rights != self.clean_rights() || self.ep != self.valid_ep() {
            return false;
        }
        for side in [1, -1] {
            let mut occupied = 0;
            let mut pawns = 0;
            let mut kings = 0;
            for (sq, &piece) in self.pieces.iter().enumerate() {
                let piece = piece * side;
                occupied += u8::from(piece > 0);
                if piece == PAWN {
                    if !(8..56).contains(&sq) {
                        return false;
                    }
                    pawns += 1;
                }
                kings += u8::from(piece == KING);
            }
            if occupied > 16 || pawns > 8 || kings != 1 {
                return false;
            }
        }
        let side = self.color();
        if self.in_check(-side) {
            return false;
        }
        let king = self.king(side).unwrap();
        let checks = self.attackers(king, -side);
        if checks.count_ones() > 2 {
            return false;
        }
        if let Some(ep) = self.ep {
            // With a raw en passant square the preceding move is known.
            let mut before = *self;
            before.pieces[offset(ep, 0, side).unwrap() as usize] = -side * PAWN;
            before.pieces[offset(ep, 0, -side).unwrap() as usize] = 0;
            return !before.in_check(side);
        }
        if checks.count_ones() == 2 {
            let first = checks.trailing_zeros() as u8;
            let second = checks.ilog2() as u8;
            let delta = |sq: u8| {
                (
                    (sq % 8) as i8 - (king % 8) as i8,
                    (sq / 8) as i8 - (king / 8) as i8,
                )
            };
            let (ax, ay) = delta(first);
            let (bx, by) = delta(second);
            if ax * by == ay * bx {
                return false;
            }
        }
        true
    }

    fn pseudo_from(&self, from: u8, mut emit: impl FnMut(Ply) -> bool) -> bool {
        let side = self.color();
        let piece = self.at(from) * side;
        let mut target = |to| {
            if piece == PAWN && !(8..56).contains(&to) {
                [QUEEN, ROOK, BISHOP, KNIGHT].into_iter().any(|promotion| {
                    emit(Ply {
                        from,
                        to,
                        promotion,
                    })
                })
            } else {
                emit(Ply {
                    from,
                    to,
                    promotion: 0,
                })
            }
        };
        if piece == PAWN {
            if let Some(to) = offset(from, 0, side).filter(|&sq| self.at(sq) == 0) {
                if target(to) {
                    return true;
                }
                if from / 8 == if self.white { 1 } else { 6 } {
                    let next = offset(to, 0, side).unwrap();
                    if self.at(next) == 0 && target(next) {
                        return true;
                    }
                }
            }
            let ep = self.valid_ep();
            for dx in [-1, 1] {
                if let Some(to) = offset(from, dx, side) {
                    let captured = self.at(to) * side;
                    if ((-KING < captured && captured < 0) || Some(to) == ep) && target(to) {
                        return true;
                    }
                }
            }
            return false;
        }
        if piece == KNIGHT {
            return JUMPS
                .into_iter()
                .filter_map(|(dx, dy)| offset(from, dx, dy))
                .any(|to| {
                    let captured = self.at(to) * side;
                    -KING < captured && captured <= 0 && target(to)
                });
        }
        if !(BISHOP..=KING).contains(&piece) {
            return false;
        }
        for (dirs, skip) in [(ORTHOGONAL, BISHOP), (DIAGONAL, ROOK)] {
            if piece == skip {
                continue;
            }
            for (dx, dy) in dirs {
                let mut cursor = from;
                while let Some(to) = offset(cursor, dx, dy) {
                    let captured = self.at(to) * side;
                    if -KING < captured && captured <= 0 && target(to) {
                        return true;
                    }
                    if captured != 0 || piece == KING {
                        break;
                    }
                    cursor = to;
                }
            }
        }
        if piece == KING {
            let home = if self.white { 0 } else { 56 };
            let rights = (self.clean_rights() >> if self.white { 0 } else { 2 }) & 3;
            if from == home + 4 && rights != 0 && self.attackers(from, -side) == 0 {
                if rights & 1 != 0
                    && self.at(home + 5) == 0
                    && self.at(home + 6) == 0
                    && self.attackers(home + 5, -side) == 0
                    && target(home + 6)
                {
                    return true;
                }
                if rights & 2 != 0
                    && (1..=3).all(|file| self.at(home + file) == 0)
                    && self.attackers(home + 3, -side) == 0
                    && target(home + 2)
                {
                    return true;
                }
            }
        }
        false
    }

    fn safe(&self, ply: Ply) -> bool {
        !self.after(ply).in_check(self.color())
    }

    pub fn legal(&self, ply: Ply) -> bool {
        self.pseudo_from(ply.from, |candidate| {
            candidate == ply && self.safe(candidate)
        })
    }

    fn visit_legal(&self, mut visit: impl FnMut(Ply) -> bool) -> bool {
        (0..64).any(|from| self.pseudo_from(from, |ply| self.safe(ply) && visit(ply)))
    }

    pub fn legal_moves(&self) -> Vec<Ply> {
        let mut moves = Vec::with_capacity(32);
        self.visit_legal(|ply| {
            moves.push(ply);
            false
        });
        moves
    }

    pub fn legal_move_count(&self) -> usize {
        let mut count = 0;
        self.visit_legal(|_| {
            count += 1;
            false
        });
        count
    }

    pub fn has_moves(&self) -> bool {
        self.visit_legal(|_| true)
    }

    pub fn after(&self, ply: Ply) -> Self {
        let mut next = *self;
        let side = self.color();
        let moved = self.at(ply.from);
        let captured = self.at(ply.to);
        next.pieces[ply.from as usize] = 0;
        next.pieces[ply.to as usize] = if ply.promotion == 0 {
            moved
        } else {
            ply.promotion * side
        };
        next.ep = None;
        next.rights = self.clean_rights();
        next.white = !self.white;
        next.fullmove = self.fullmove.saturating_add(u32::from(!self.white));
        next.halfmove = if moved.abs() == PAWN || captured != 0 {
            0
        } else {
            self.halfmove.saturating_add(1)
        };
        if moved.abs() == PAWN {
            if ply.from % 8 != ply.to % 8 && captured == 0 {
                next.pieces[offset(ply.to, 0, -side).unwrap() as usize] = 0;
            }
            if ply.from.abs_diff(ply.to) == 16 {
                next.ep = offset(ply.from, 0, side);
            }
        }
        if moved.abs() == KING {
            next.rights &= if self.white { 12 } else { 3 };
            if ply.from.abs_diff(ply.to) == 2 {
                let (rook, target) = if ply.to > ply.from {
                    (ply.to + 1, ply.to - 1)
                } else {
                    (ply.to - 2, ply.to + 1)
                };
                next.pieces[rook as usize] = 0;
                next.pieces[target as usize] = side * ROOK;
            }
        }
        for (sq, mask) in [(7, 1), (0, 2), (63, 4), (56, 8)] {
            if ply.from == sq || ply.to == sq {
                next.rights &= !mask;
            }
        }
        next
    }

    fn legal_ep(&self) -> Option<u8> {
        let to = self.valid_ep()?;
        let side = self.color();
        [-1, 1]
            .into_iter()
            .filter_map(|dx| offset(to, dx, -side))
            .any(|from| {
                self.at(from) == side * PAWN
                    && self.safe(Ply {
                        from,
                        to,
                        promotion: 0,
                    })
            })
            .then_some(to)
    }

    pub fn key(&self) -> PositionKey {
        PositionKey {
            pieces: self.pieces,
            white: self.white,
            rights: self.clean_rights(),
            ep: self.legal_ep(),
        }
    }

    pub fn insufficient_material(&self) -> bool {
        let mut knights = 0;
        let mut bishops = 0;
        for (sq, &piece) in self.pieces.iter().enumerate() {
            match piece.abs() {
                PAWN | ROOK | QUEEN => return false,
                KNIGHT => knights += 1,
                BISHOP => bishops |= 1 << ((sq % 8 + sq / 8) % 2),
                _ => {}
            }
        }
        (knights == 0 && bishops != 3) || (knights == 1 && bishops == 0)
    }

    pub fn san(&self, ply: Ply) -> String {
        let piece = self.at(ply.from).abs();
        let next = self.after(ply);
        let mut name = String::with_capacity(8);
        if piece == KING && ply.from.abs_diff(ply.to) == 2 {
            name.push_str(if ply.to > ply.from { "O-O" } else { "O-O-O" });
        } else {
            let source = square_name(ply.from);
            let capture = self.at(ply.to) != 0 || (piece == PAWN && ply.from % 8 != ply.to % 8);
            if piece != PAWN {
                name.push(symbol(piece));
                if piece != KING {
                    let mut ambiguous = false;
                    let mut file = false;
                    let mut rank = false;
                    for from in 0..64 {
                        if from != ply.from
                            && self.at(from) == self.at(ply.from)
                            && self.legal(Ply { from, ..ply })
                        {
                            ambiguous = true;
                            file |= from % 8 == ply.from % 8;
                            rank |= from / 8 == ply.from / 8;
                        }
                    }
                    if ambiguous {
                        if !file || rank {
                            name.push(source[0]);
                        }
                        if file {
                            name.push(source[1]);
                        }
                    }
                }
            } else if capture {
                name.push(source[0]);
            }
            if capture {
                name.push('x');
            }
            name.extend(square_name(ply.to));
            if ply.promotion != 0 {
                name.push('=');
                name.push(symbol(ply.promotion));
            }
        }
        if next.is_check() {
            name.push(if next.has_moves() { '+' } else { '#' });
        }
        name
    }

    pub fn parse_san(&self, text: &str) -> Result<Ply, &'static str> {
        let token = text.trim_end_matches(['+', '#', '!', '?']);
        if matches!(token, "O-O" | "0-0" | "O-O-O" | "0-0-0") {
            let from = if self.white { 4 } else { 60 };
            let to = if token.len() == 3 { from + 2 } else { from - 2 };
            let ply = Ply {
                from,
                to,
                promotion: 0,
            };
            return (self.at(from) == KING * self.color() && self.legal(ply))
                .then_some(ply)
                .ok_or("Illegal castling");
        }
        let mut bytes = token.as_bytes();
        let promotion = match bytes.last() {
            Some(last @ (b'Q' | b'R' | b'B' | b'N' | b'q' | b'r' | b'b' | b'n')) => {
                let kind = piece_index(*last).unwrap().abs();
                bytes = &bytes[..bytes.len() - 1];
                bytes = bytes.strip_suffix(b"=").unwrap_or(bytes);
                kind
            }
            _ => 0,
        };
        let prefix_len = bytes.len().checked_sub(2).ok_or("Invalid SAN")?;
        let to = square(&bytes[prefix_len..]).ok_or("Invalid SAN destination")?;
        let prefix = &bytes[..prefix_len];
        let (piece, prefix) = match prefix {
            [name @ (b'K' | b'Q' | b'R' | b'B' | b'N' | b'P'), tail @ ..] => {
                (piece_index(*name), tail)
            }
            _ => (None, prefix),
        };
        let hint = match prefix.last() {
            Some(b'x' | b'-') => &prefix[..prefix.len() - 1],
            _ => prefix,
        };
        let (file, rank, from_square) = match hint {
            [] => (None, None, None),
            [file @ b'a'..=b'h'] => (Some(file - b'a'), None, None),
            [rank @ b'1'..=b'8'] => (None, Some(rank - b'1'), None),
            [_, _] => (None, None, Some(square(hint).ok_or("Invalid SAN origin")?)),
            _ => return Err("Invalid SAN origin"),
        };
        let kind = piece.unwrap_or(PAWN);
        let file = file.or_else(|| (kind == PAWN && from_square.is_none()).then_some(to % 8));
        let must_move = kind * self.color();
        let mut found = None;
        let ambiguous = (0..64).any(|from| {
            if (piece.is_some() || from_square.is_none()) && self.at(from) != must_move
                || from_square.is_some_and(|sq| from != sq)
                || file.is_some_and(|f| from % 8 != f)
                || rank.is_some_and(|r| from / 8 != r)
            {
                return false;
            }
            let ply = Ply {
                from,
                to,
                promotion,
            };
            if self.legal(ply) {
                return found.replace(ply).is_some();
            }
            false
        });
        if ambiguous {
            return Err("Ambiguous SAN");
        }
        found.ok_or("Illegal SAN move")
    }
}

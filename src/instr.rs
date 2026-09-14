//! Decodes a function body's raw instruction bytes (`Code::body`) into a flat
//! `Vec<Instr>`.
//!
//! This does not build a tree out of `block`/`loop`/`if`/`else`/`end` - those
//! stay as individual instructions in the flat sequence, the same shape the
//! binary format itself uses. Matching a `br` or `br_if` target to the block
//! it exits, and matching an `if` to its `else`, is the interpreter's problem
//! to solve while it walks the sequence, not something resolved up front
//! here. What this module guarantees is narrower: every opcode is one this
//! crate supports, every immediate decoded cleanly, and nothing was left
//! unread at the end of the body.

use crate::leb::{self, LebError};
use crate::types::ValType;

/// A block's signature: what it leaves on the stack when it completes.
///
/// The binary format also allows a block type to be a type index into the
/// module's type section, for blocks with more than one result (the
/// multi-value proposal). This crate does not run those - see the README's
/// "Not implemented" list - so `read_block_type` reports any such index as
/// `DecodeErrorKind::UnsupportedBlockType` rather than a variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    Empty,
    Value(ValType),
}

/// One decoded instruction. Variants map one-to-one to the opcodes listed in
/// the README's "The interpreter executes" table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instr {
    Unreachable,
    Nop,
    Block(BlockType),
    Loop(BlockType),
    If(BlockType),
    Else,
    End,
    Br(u32),
    BrIf(u32),
    Return,
    Call(u32),

    Drop,
    Select,

    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),

    I32Const(i32),
    I64Const(i64),

    I32Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,
    I32Clz,
    I32Ctz,
    I32Popcnt,
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rotl,
    I32Rotr,

    I64Eqz,
    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    I64Clz,
    I64Ctz,
    I64Popcnt,
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,

    I32WrapI64,
    I64ExtendI32S,
    I64ExtendI32U,
}

/// Why decoding a function body failed, and where.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeError {
    /// The byte offset within the body that broke.
    pub offset: usize,
    pub kind: DecodeErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeErrorKind {
    /// An immediate (a label depth, a local index, a const value, a block
    /// type) failed to decode as LEB128.
    Leb(LebError),
    /// A byte that was supposed to open an instruction is not one this crate
    /// implements - either not part of the WebAssembly MVP opcode set, or
    /// part of it but outside what this interpreter runs (memory, globals,
    /// `call_indirect`, `br_table`, floats).
    UnsupportedOpcode(u8),
    /// A block type named a type index rather than empty or a single value
    /// type. That is only meaningful for the multi-value proposal, which
    /// this crate does not implement.
    UnsupportedBlockType,
}

impl std::fmt::Display for DecodeErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeErrorKind::Leb(e) => write!(f, "{e}"),
            DecodeErrorKind::UnsupportedOpcode(byte) => write!(f, "unsupported opcode {byte:#04x}"),
            DecodeErrorKind::UnsupportedBlockType => f.write_str("unsupported block type (multi-value)"),
        }
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "at instruction byte {}: {}", self.offset, self.kind)
    }
}

impl std::error::Error for DecodeError {}

fn read_u32(body: &[u8], pos: &mut usize) -> Result<u32, DecodeError> {
    let offset = *pos;
    leb::read_u32(body, pos).map_err(|e| DecodeError { offset, kind: DecodeErrorKind::Leb(e) })
}

fn read_i32(body: &[u8], pos: &mut usize) -> Result<i32, DecodeError> {
    let offset = *pos;
    leb::read_i32(body, pos).map_err(|e| DecodeError { offset, kind: DecodeErrorKind::Leb(e) })
}

fn read_i64(body: &[u8], pos: &mut usize) -> Result<i64, DecodeError> {
    let offset = *pos;
    leb::read_i64(body, pos).map_err(|e| DecodeError { offset, kind: DecodeErrorKind::Leb(e) })
}

/// Decodes a block type: `0x40` (empty), a value type byte (a single
/// result), or an `s33` type index (unsupported here). All three share one
/// encoding space, so this reads a 33-bit signed LEB128 and only afterward
/// sorts out which case it was.
fn read_block_type(body: &[u8], pos: &mut usize) -> Result<BlockType, DecodeError> {
    let offset = *pos;
    let value = leb::read_signed(body, pos, 33).map_err(|e| DecodeError { offset, kind: DecodeErrorKind::Leb(e) })?;
    match value {
        -64 => Ok(BlockType::Empty),
        -1 => Ok(BlockType::Value(ValType::I32)),
        -2 => Ok(BlockType::Value(ValType::I64)),
        -3 => Ok(BlockType::Value(ValType::F32)),
        -4 => Ok(BlockType::Value(ValType::F64)),
        _ => Err(DecodeError { offset, kind: DecodeErrorKind::UnsupportedBlockType }),
    }
}

/// Decodes every instruction in `body` (a `Code::body`, including its
/// trailing `end`) into a flat sequence.
pub fn decode(body: &[u8]) -> Result<Vec<Instr>, DecodeError> {
    let mut pos = 0;
    let mut instrs = Vec::new();

    while pos < body.len() {
        let opcode_offset = pos;
        let opcode = body[pos];
        pos += 1;

        let instr = match opcode {
            0x00 => Instr::Unreachable,
            0x01 => Instr::Nop,
            0x02 => Instr::Block(read_block_type(body, &mut pos)?),
            0x03 => Instr::Loop(read_block_type(body, &mut pos)?),
            0x04 => Instr::If(read_block_type(body, &mut pos)?),
            0x05 => Instr::Else,
            0x0B => Instr::End,
            0x0C => Instr::Br(read_u32(body, &mut pos)?),
            0x0D => Instr::BrIf(read_u32(body, &mut pos)?),
            0x0F => Instr::Return,
            0x10 => Instr::Call(read_u32(body, &mut pos)?),

            0x1A => Instr::Drop,
            0x1B => Instr::Select,

            0x20 => Instr::LocalGet(read_u32(body, &mut pos)?),
            0x21 => Instr::LocalSet(read_u32(body, &mut pos)?),
            0x22 => Instr::LocalTee(read_u32(body, &mut pos)?),

            0x41 => Instr::I32Const(read_i32(body, &mut pos)?),
            0x42 => Instr::I64Const(read_i64(body, &mut pos)?),

            0x45 => Instr::I32Eqz,
            0x46 => Instr::I32Eq,
            0x47 => Instr::I32Ne,
            0x48 => Instr::I32LtS,
            0x49 => Instr::I32LtU,
            0x4A => Instr::I32GtS,
            0x4B => Instr::I32GtU,
            0x4C => Instr::I32LeS,
            0x4D => Instr::I32LeU,
            0x4E => Instr::I32GeS,
            0x4F => Instr::I32GeU,

            0x50 => Instr::I64Eqz,
            0x51 => Instr::I64Eq,
            0x52 => Instr::I64Ne,
            0x53 => Instr::I64LtS,
            0x54 => Instr::I64LtU,
            0x55 => Instr::I64GtS,
            0x56 => Instr::I64GtU,
            0x57 => Instr::I64LeS,
            0x58 => Instr::I64LeU,
            0x59 => Instr::I64GeS,
            0x5A => Instr::I64GeU,

            0x67 => Instr::I32Clz,
            0x68 => Instr::I32Ctz,
            0x69 => Instr::I32Popcnt,
            0x6A => Instr::I32Add,
            0x6B => Instr::I32Sub,
            0x6C => Instr::I32Mul,
            0x6D => Instr::I32DivS,
            0x6E => Instr::I32DivU,
            0x6F => Instr::I32RemS,
            0x70 => Instr::I32RemU,
            0x71 => Instr::I32And,
            0x72 => Instr::I32Or,
            0x73 => Instr::I32Xor,
            0x74 => Instr::I32Shl,
            0x75 => Instr::I32ShrS,
            0x76 => Instr::I32ShrU,
            0x77 => Instr::I32Rotl,
            0x78 => Instr::I32Rotr,

            0x79 => Instr::I64Clz,
            0x7A => Instr::I64Ctz,
            0x7B => Instr::I64Popcnt,
            0x7C => Instr::I64Add,
            0x7D => Instr::I64Sub,
            0x7E => Instr::I64Mul,
            0x7F => Instr::I64DivS,
            0x80 => Instr::I64DivU,
            0x81 => Instr::I64RemS,
            0x82 => Instr::I64RemU,
            0x83 => Instr::I64And,
            0x84 => Instr::I64Or,
            0x85 => Instr::I64Xor,
            0x86 => Instr::I64Shl,
            0x87 => Instr::I64ShrS,
            0x88 => Instr::I64ShrU,
            0x89 => Instr::I64Rotl,
            0x8A => Instr::I64Rotr,

            0xA7 => Instr::I32WrapI64,
            0xAC => Instr::I64ExtendI32S,
            0xAD => Instr::I64ExtendI32U,

            other => {
                return Err(DecodeError { offset: opcode_offset, kind: DecodeErrorKind::UnsupportedOpcode(other) });
            }
        };
        instrs.push(instr);
    }

    Ok(instrs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_square_function_body() {
        // local.get 0, local.get 0, i32.mul, end - the body from the
        // README's `square` example.
        let body = [0x20, 0x00, 0x20, 0x00, 0x6C, 0x0B];
        assert_eq!(
            decode(&body),
            Ok(vec![Instr::LocalGet(0), Instr::LocalGet(0), Instr::I32Mul, Instr::End])
        );
    }

    #[test]
    fn decodes_consts_and_arithmetic() {
        let body = [0x41, 0x7F, 0x42, 0x80, 0x01, 0x6A, 0x0B]; // i32.const -1, i64.const 128, i32.add, end
        assert_eq!(
            decode(&body),
            Ok(vec![Instr::I32Const(-1), Instr::I64Const(128), Instr::I32Add, Instr::End])
        );
    }

    #[test]
    fn decodes_nested_block_loop_if_else_with_block_types() {
        // block (i32) loop i32.const 0 if else end end end
        let body = [
            0x02, 0x7F, // block (result i32)
            0x03, 0x40, // loop
            0x41, 0x00, // i32.const 0
            0x04, 0x40, // if
            0x05, // else
            0x0B, // end (if)
            0x0B, // end (loop)
            0x0B, // end (block)
        ];
        assert_eq!(
            decode(&body),
            Ok(vec![
                Instr::Block(BlockType::Value(ValType::I32)),
                Instr::Loop(BlockType::Empty),
                Instr::I32Const(0),
                Instr::If(BlockType::Empty),
                Instr::Else,
                Instr::End,
                Instr::End,
                Instr::End,
            ])
        );
    }

    #[test]
    fn decodes_branches_and_calls() {
        let body = [0x0C, 0x02, 0x0D, 0x01, 0x10, 0x03, 0x0F, 0x0B]; // br 2, br_if 1, call 3, return, end
        assert_eq!(
            decode(&body),
            Ok(vec![Instr::Br(2), Instr::BrIf(1), Instr::Call(3), Instr::Return, Instr::End])
        );
    }

    #[test]
    fn rejects_an_unsupported_opcode() {
        let body = [0x01, 0x28, 0x0B]; // nop, i32.load (no memory support), end
        assert_eq!(
            decode(&body),
            Err(DecodeError { offset: 1, kind: DecodeErrorKind::UnsupportedOpcode(0x28) })
        );
    }

    #[test]
    fn rejects_a_multi_value_block_type() {
        let body = [0x02, 0x00, 0x0B, 0x0B]; // block (type 0) end end - type index, not empty/value
        assert_eq!(
            decode(&body),
            Err(DecodeError { offset: 1, kind: DecodeErrorKind::UnsupportedBlockType })
        );
    }

    #[test]
    fn rejects_a_truncated_immediate() {
        let body = [0x20]; // local.get with no index byte
        assert_eq!(
            decode(&body),
            Err(DecodeError { offset: 1, kind: DecodeErrorKind::Leb(LebError::UnexpectedEof) })
        );
    }
}

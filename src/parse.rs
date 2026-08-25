//! The WebAssembly binary format decoder.
//!
//! This module builds up the module structure one piece at a time: header
//! first, then sections. After the header, a binary is a flat sequence of
//! `(id, size, content)` sections; `parse` walks that sequence once, checking
//! the ordering rule (known section ids must strictly increase; custom
//! sections are exempt and may appear anywhere, any number of times) and
//! handing each section's bytes to whatever understands that id. Only the
//! type section has a real parser so far - everything else with a known id
//! is skipped by length and recorded in `skipped_sections`, which is also
//! where table/memory/global/element/data/data-count sections will stay
//! permanently, since this crate never gives guest code memory or tables.

use std::fmt;

use crate::leb::{self, LebError};
use crate::types::{FuncType, ValType};

/// The maximum section id defined by the binary format (the data-count
/// section added for bulk memory). Anything past this is not a section this
/// format knows about, known or otherwise.
const MAX_KNOWN_SECTION_ID: u8 = 12;

/// The byte that opens every function type: `0x60`.
const FUNC_TYPE_TAG: u8 = 0x60;

/// A decoded WebAssembly module.
///
/// Fields fill in as the parser grows; right now only `types` reflects real
/// section content; imports, functions, exports, start and code all arrive
/// as their sections get parsers of their own.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Module {
    /// The type section: every function signature the module declares or
    /// refers to, indexed by type index.
    pub types: Vec<FuncType>,
    /// The name of each custom section, in the order it appeared. Custom
    /// section contents are not otherwise interpreted.
    pub custom_sections: Vec<String>,
    /// The id of each known, non-custom section that was skipped rather than
    /// parsed, in the order it appeared.
    pub skipped_sections: Vec<u8>,
}

/// The four bytes that open every WebAssembly binary: `\0asm`.
pub const MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6D];

/// The only version this crate understands. There has only ever been one
/// released version of the binary format; a `2` or later here would mean a
/// future format this parser was not written against.
pub const VERSION: [u8; 4] = [0x01, 0x00, 0x00, 0x00];

/// Why parsing failed, and where.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseError {
    /// The byte offset that broke.
    pub offset: usize,
    pub kind: ParseErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseErrorKind {
    /// The first four bytes are not `\0asm`.
    NotWasm,
    /// The magic number is fine but the version is not one this parser
    /// implements.
    UnsupportedVersion,
    /// The input ended before a required byte.
    UnexpectedEof,
    /// A LEB128 immediate (a section size, a vector count, a type tag's
    /// operand) failed to decode.
    Leb(LebError),
    /// A section id past `MAX_KNOWN_SECTION_ID`.
    UnknownSectionId(u8),
    /// A known section appeared out of the order the format requires, or a
    /// second copy of a section id that may only appear once. Custom
    /// sections (id 0) are exempt from this check.
    SectionOutOfOrder,
    /// A section's declared byte length did not match the number of bytes
    /// its content actually decoded to.
    SectionSizeMismatch,
    /// A byte that was supposed to encode a value type is none of `i32`,
    /// `i64`, `f32`, `f64`.
    InvalidValType(u8),
    /// A function type did not start with `0x60`.
    InvalidFuncType(u8),
    /// A custom section's name was not valid UTF-8.
    InvalidUtf8,
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::NotWasm => f.write_str("not a WebAssembly binary (bad magic number)"),
            ParseErrorKind::UnsupportedVersion => f.write_str("unsupported WebAssembly version"),
            ParseErrorKind::UnexpectedEof => f.write_str("unexpected end of input"),
            ParseErrorKind::Leb(e) => write!(f, "{e}"),
            ParseErrorKind::UnknownSectionId(id) => write!(f, "unknown section id {id}"),
            ParseErrorKind::SectionOutOfOrder => f.write_str("section out of order"),
            ParseErrorKind::SectionSizeMismatch => {
                f.write_str("section content did not match its declared size")
            }
            ParseErrorKind::InvalidValType(byte) => {
                write!(f, "invalid value type byte {byte:#04x}")
            }
            ParseErrorKind::InvalidFuncType(byte) => {
                write!(f, "invalid function type tag {byte:#04x} (expected 0x60)")
            }
            ParseErrorKind::InvalidUtf8 => f.write_str("custom section name is not valid UTF-8"),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.offset, self.kind)
    }
}

impl std::error::Error for ParseError {}

/// Checks the magic number and version at the start of `bytes` and returns
/// the offset of the first byte after them, i.e. where section parsing would
/// continue from.
///
/// A mismatched magic byte is reported as `NotWasm` at offset `0` rather than
/// at the exact byte that differed - the header is one indivisible thing, and
/// "this is not a wasm file" is a more useful message than the position of
/// the first wrong byte within it.
pub fn parse_header(bytes: &[u8]) -> Result<usize, ParseError> {
    for (i, &expected) in MAGIC.iter().enumerate() {
        match bytes.get(i) {
            Some(&byte) if byte == expected => {}
            Some(_) => return Err(ParseError { offset: 0, kind: ParseErrorKind::NotWasm }),
            None => return Err(ParseError { offset: i, kind: ParseErrorKind::UnexpectedEof }),
        }
    }
    for (i, &expected) in VERSION.iter().enumerate() {
        let offset = MAGIC.len() + i;
        match bytes.get(offset) {
            Some(&byte) if byte == expected => {}
            Some(_) => {
                return Err(ParseError { offset: MAGIC.len(), kind: ParseErrorKind::UnsupportedVersion })
            }
            None => return Err(ParseError { offset, kind: ParseErrorKind::UnexpectedEof }),
        }
    }
    Ok(MAGIC.len() + VERSION.len())
}

/// Parses a complete WebAssembly binary: the header, then every section.
///
/// Sections are read as `(id: u8, size: u32, content: [u8; size])` in a flat
/// loop. Known section ids (1 through `MAX_KNOWN_SECTION_ID`) must strictly
/// increase from one section to the next, which also forbids repeating one;
/// custom sections (id 0) are exempt and may appear anywhere, any number of
/// times. Only the type section is actually decoded; every other known id is
/// skipped by length and recorded in `Module::skipped_sections`.
pub fn parse(bytes: &[u8]) -> Result<Module, ParseError> {
    let mut pos = parse_header(bytes)?;
    let mut module = Module::default();
    let mut last_known_id: Option<u8> = None;

    while pos < bytes.len() {
        let section_start = pos;
        let id = bytes[pos];
        pos += 1;

        let size = leb::read_u32(bytes, &mut pos)
            .map_err(|e| ParseError { offset: pos, kind: ParseErrorKind::Leb(e) })?
            as usize;
        let body_start = pos;
        let body_end = body_start
            .checked_add(size)
            .filter(|&end| end <= bytes.len())
            .ok_or(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })?;
        let body = &bytes[body_start..body_end];

        match id {
            0 => {
                let mut name_pos = 0;
                let name = read_name(body, &mut name_pos, body_start)?;
                module.custom_sections.push(name);
            }
            1..=MAX_KNOWN_SECTION_ID => {
                if last_known_id.is_some_and(|last| id <= last) {
                    return Err(ParseError { offset: section_start, kind: ParseErrorKind::SectionOutOfOrder });
                }
                last_known_id = Some(id);
                if id == 1 {
                    module.types = parse_type_section(body, body_start)?;
                } else {
                    module.skipped_sections.push(id);
                }
            }
            other => {
                return Err(ParseError { offset: section_start, kind: ParseErrorKind::UnknownSectionId(other) });
            }
        }

        pos = body_end;
    }

    Ok(module)
}

/// Decodes a `u32` from `bytes` at `*pos`, translating a LEB128 failure into
/// a `ParseError` whose offset is `base` (the absolute position of `bytes`
/// within the whole module) plus the position the decoder stopped at.
fn read_u32_at(bytes: &[u8], pos: &mut usize, base: usize) -> Result<u32, ParseError> {
    leb::read_u32(bytes, pos).map_err(|e| ParseError { offset: base + *pos, kind: ParseErrorKind::Leb(e) })
}

/// Decodes the type section: `vec(functype)`, where each `functype` is
/// `0x60 vec(valtype) vec(valtype)` (parameters, then results).
fn parse_type_section(body: &[u8], base: usize) -> Result<Vec<FuncType>, ParseError> {
    let mut pos = 0;
    let count = read_u32_at(body, &mut pos, base)?;
    let mut types = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let tag_offset = base + pos;
        let tag = *body
            .get(pos)
            .ok_or(ParseError { offset: tag_offset, kind: ParseErrorKind::UnexpectedEof })?;
        pos += 1;
        if tag != FUNC_TYPE_TAG {
            return Err(ParseError { offset: tag_offset, kind: ParseErrorKind::InvalidFuncType(tag) });
        }
        let params = read_val_type_vec(body, &mut pos, base)?;
        let results = read_val_type_vec(body, &mut pos, base)?;
        types.push(FuncType { params, results });
    }

    // The type section is the one section this parser fully understands, so
    // it is also the one place it can catch a size field that lied: if the
    // declared count of types did not consume exactly `size` bytes, the file
    // is malformed even though every individual value decoded cleanly.
    if pos != body.len() {
        return Err(ParseError { offset: base + pos, kind: ParseErrorKind::SectionSizeMismatch });
    }

    Ok(types)
}

fn read_val_type_vec(body: &[u8], pos: &mut usize, base: usize) -> Result<Vec<ValType>, ParseError> {
    let count = read_u32_at(body, pos, base)?;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let offset = base + *pos;
        let byte = *body.get(*pos).ok_or(ParseError { offset, kind: ParseErrorKind::UnexpectedEof })?;
        *pos += 1;
        out.push(ValType::from_byte(byte).ok_or(ParseError { offset, kind: ParseErrorKind::InvalidValType(byte) })?);
    }
    Ok(out)
}

/// Decodes a custom section's name: `vec(byte)` interpreted as UTF-8. Any
/// bytes after the name (the section's actual payload) are left unread -
/// custom section contents are not otherwise interpreted.
fn read_name(body: &[u8], pos: &mut usize, base: usize) -> Result<String, ParseError> {
    let len = read_u32_at(body, pos, base)? as usize;
    let start = *pos;
    let end = start
        .checked_add(len)
        .filter(|&end| end <= body.len())
        .ok_or(ParseError { offset: base + body.len(), kind: ParseErrorKind::UnexpectedEof })?;
    let name = std::str::from_utf8(&body[start..end])
        .map_err(|_| ParseError { offset: base + start, kind: ParseErrorKind::InvalidUtf8 })?
        .to_string();
    *pos = end;
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_bare_header() {
        let bytes = [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        assert_eq!(parse_header(&bytes), Ok(8));
    }

    #[test]
    fn ignores_trailing_bytes() {
        let bytes = [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF];
        assert_eq!(parse_header(&bytes), Ok(8));
    }

    #[test]
    fn rejects_wrong_magic() {
        let bytes = [0x00, 0x61, 0x73, 0x00, 0x01, 0x00, 0x00, 0x00];
        assert_eq!(
            parse_header(&bytes),
            Err(ParseError { offset: 0, kind: ParseErrorKind::NotWasm })
        );
    }

    #[test]
    fn rejects_unsupported_version() {
        let bytes = [0x00, 0x61, 0x73, 0x6D, 0x02, 0x00, 0x00, 0x00];
        assert_eq!(
            parse_header(&bytes),
            Err(ParseError { offset: 4, kind: ParseErrorKind::UnsupportedVersion })
        );
    }

    #[test]
    fn reports_truncation_at_the_exact_missing_byte() {
        assert_eq!(
            parse_header(&[]),
            Err(ParseError { offset: 0, kind: ParseErrorKind::UnexpectedEof })
        );
        assert_eq!(
            parse_header(&[0x00, 0x61, 0x73, 0x6D]),
            Err(ParseError { offset: 4, kind: ParseErrorKind::UnexpectedEof })
        );
        assert_eq!(
            parse_header(&[0x00, 0x61, 0x73, 0x6D, 0x01, 0x00]),
            Err(ParseError { offset: 6, kind: ParseErrorKind::UnexpectedEof })
        );
    }

    #[test]
    fn a_truncated_magic_number_is_eof_not_a_bad_magic() {
        // The first three bytes match \0asm; there is no fourth byte to
        // compare, so this is a truncation, not a content mismatch.
        assert_eq!(
            parse_header(&[0x00, 0x61, 0x73]),
            Err(ParseError { offset: 3, kind: ParseErrorKind::UnexpectedEof })
        );
    }

    fn header() -> Vec<u8> {
        vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]
    }

    #[test]
    fn parses_a_module_with_no_sections() {
        assert_eq!(parse(&header()), Ok(Module::default()));
    }

    #[test]
    fn parses_a_type_section() {
        let mut bytes = header();
        // type section: id 1, size 6, one functype (i32) -> i32
        bytes.extend_from_slice(&[0x01, 0x06, 0x01, 0x60, 0x01, 0x7F, 0x01, 0x7F]);
        let module = parse(&bytes).unwrap();
        assert_eq!(
            module.types,
            vec![FuncType { params: vec![ValType::I32], results: vec![ValType::I32] }]
        );
    }

    #[test]
    fn parses_multiple_func_types_including_no_params_and_no_results() {
        let mut bytes = header();
        bytes.extend_from_slice(&[
            0x01, // type section id
            0x0A, // size
            0x02, // two functypes
            0x60, 0x00, 0x00, // () -> ()
            0x60, 0x02, 0x7F, 0x7E, 0x01, 0x7C, // (i32, i64) -> f64
        ]);
        let module = parse(&bytes).unwrap();
        assert_eq!(
            module.types,
            vec![
                FuncType { params: vec![], results: vec![] },
                FuncType {
                    params: vec![ValType::I32, ValType::I64],
                    results: vec![ValType::F64],
                },
            ]
        );
    }

    #[test]
    fn records_custom_section_names_and_ignores_their_content() {
        let mut bytes = header();
        // custom section: id 0, size 4, name "x" (len 1) followed by 2 bytes
        // of payload this parser never looks at
        bytes.extend_from_slice(&[0x00, 0x04, 0x01, b'x', 0xAA, 0xBB]);
        let module = parse(&bytes).unwrap();
        assert_eq!(module.custom_sections, vec!["x".to_string()]);
        assert!(module.types.is_empty());
    }

    #[test]
    fn skips_known_sections_it_does_not_parse_yet() {
        let mut bytes = header();
        // memory section: id 5, size 3, contents irrelevant - not decoded
        bytes.extend_from_slice(&[0x05, 0x03, 0x00, 0x01, 0x00]);
        let module = parse(&bytes).unwrap();
        assert_eq!(module.skipped_sections, vec![5]);
    }

    #[test]
    fn custom_sections_may_appear_anywhere_and_repeat() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x00, 0x02, 0x01, b'a']); // custom "a"
        bytes.extend_from_slice(&[0x01, 0x04, 0x01, 0x60, 0x00, 0x00]); // type section
        bytes.extend_from_slice(&[0x00, 0x02, 0x01, b'b']); // custom "b"
        let module = parse(&bytes).unwrap();
        assert_eq!(module.custom_sections, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(module.types, vec![FuncType { params: vec![], results: vec![] }]);
    }

    #[test]
    fn rejects_an_unknown_section_id() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x0D, 0x00]); // id 13 does not exist
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 8, kind: ParseErrorKind::UnknownSectionId(13) })
        );
    }

    #[test]
    fn rejects_sections_out_of_order() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x05, 0x00]); // memory section (id 5)
        let memory_end = bytes.len();
        bytes.extend_from_slice(&[0x01, 0x01, 0x00]); // type section (id 1), after id 5
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: memory_end, kind: ParseErrorKind::SectionOutOfOrder })
        );
    }

    #[test]
    fn rejects_a_repeated_section_id() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x01, 0x01, 0x00]); // empty type section
        let second_start = bytes.len();
        bytes.extend_from_slice(&[0x01, 0x01, 0x00]); // a second one
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: second_start, kind: ParseErrorKind::SectionOutOfOrder })
        );
    }

    #[test]
    fn rejects_a_section_that_claims_more_bytes_than_the_file_has() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x01, 0x05, 0x01, 0x60, 0x00]); // says 5 bytes, has 3
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: bytes.len(), kind: ParseErrorKind::UnexpectedEof })
        );
    }

    #[test]
    fn rejects_a_type_section_whose_size_does_not_match_its_content() {
        let mut bytes = header();
        // one functype that decodes in 6 bytes, but the section claims 7
        bytes.extend_from_slice(&[0x01, 0x07, 0x01, 0x60, 0x01, 0x7F, 0x01, 0x7F, 0x00]);
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 8 + 8, kind: ParseErrorKind::SectionSizeMismatch })
        );
    }

    #[test]
    fn rejects_a_func_type_with_the_wrong_tag() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x01, 0x03, 0x01, 0x61, 0x00]); // 0x61, not 0x60
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 8 + 3, kind: ParseErrorKind::InvalidFuncType(0x61) })
        );
    }

    #[test]
    fn rejects_an_invalid_val_type() {
        let mut bytes = header();
        // (funcref) -> (), 0x70 is a valid reftype but not a valtype this
        // crate accepts
        bytes.extend_from_slice(&[0x01, 0x04, 0x01, 0x60, 0x01, 0x70]);
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 8 + 5, kind: ParseErrorKind::InvalidValType(0x70) })
        );
    }

    #[test]
    fn rejects_a_non_utf8_custom_section_name() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x00, 0x02, 0x01, 0xFF]); // 0xFF is not valid UTF-8
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: 8 + 3, kind: ParseErrorKind::InvalidUtf8 })
        );
    }
}

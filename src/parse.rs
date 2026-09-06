//! The WebAssembly binary format decoder.
//!
//! This module builds up the module structure one piece at a time: header
//! first, then sections. After the header, a binary is a flat sequence of
//! `(id, size, content)` sections; `parse` walks that sequence once, checking
//! the ordering rule (known section ids must strictly increase; custom
//! sections are exempt and may appear anywhere, any number of times) and
//! handing each section's bytes to whatever understands that id. Type,
//! import, function, export and start sections have real parsers; the code
//! section is still skipped by length for now, and table/memory/global/
//! element/data/data-count sections will stay in `skipped_sections`
//! permanently, since this crate never gives guest code memory or tables.

use std::fmt;

use crate::leb::{self, LebError};
use crate::types::{Export, ExternKind, Func, FuncType, Import, ImportDesc, ValType};

/// The maximum section id defined by the binary format (the data-count
/// section added for bulk memory). Anything past this is not a section this
/// format knows about, known or otherwise.
const MAX_KNOWN_SECTION_ID: u8 = 12;

/// The byte that opens every function type: `0x60`.
const FUNC_TYPE_TAG: u8 = 0x60;

/// A decoded WebAssembly module.
///
/// Fields fill in as the parser grows; code section bodies (locals and
/// instruction bytes) are still missing, so `funcs` only carries each local
/// function's type index so far.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Module {
    /// The type section: every function signature the module declares or
    /// refers to, indexed by type index.
    pub types: Vec<FuncType>,
    /// The import section, in declaration order.
    pub imports: Vec<Import>,
    /// The function section: the type index of each locally defined
    /// function, in declaration order. Function indices count imported
    /// functions first, then these.
    pub funcs: Vec<Func>,
    /// The export section, in declaration order.
    pub exports: Vec<Export>,
    /// The start function's index, if the module declares one.
    pub start: Option<u32>,
    /// The name of each custom section, in the order it appeared. Custom
    /// section contents are not otherwise interpreted.
    pub custom_sections: Vec<String>,
    /// The id of each known, non-custom section that was skipped rather than
    /// parsed, in the order it appeared.
    pub skipped_sections: Vec<u8>,
}

impl Module {
    /// How many entries at the start of the function index space are
    /// imports rather than locally defined functions.
    pub fn imported_func_count(&self) -> usize {
        self.imports.iter().filter(|i| matches!(i.desc, ImportDesc::Func(_))).count()
    }

    /// The signature of the function at `index` in the function index space
    /// (imported functions first, then local ones), if `index` names a
    /// function and its type index is in range.
    pub fn func_type(&self, index: u32) -> Option<&FuncType> {
        let index = index as usize;
        let imported = self.imported_func_count();
        let type_idx = if index < imported {
            self.imports
                .iter()
                .filter_map(|i| match i.desc {
                    ImportDesc::Func(t) => Some(t),
                    _ => None,
                })
                .nth(index)?
        } else {
            self.funcs.get(index - imported)?.type_idx
        };
        self.types.get(type_idx as usize)
    }

    /// The function index exported under `name`, if `name` names a function
    /// export.
    pub fn export_func(&self, name: &str) -> Option<u32> {
        self.exports
            .iter()
            .find(|e| e.kind == ExternKind::Func && e.name == name)
            .map(|e| e.index)
    }

    /// One human-readable line per export, in declaration order, e.g.
    /// `"func square: (i32) -> i32"`.
    pub fn describe_exports(&self) -> Vec<String> {
        self.exports
            .iter()
            .map(|e| match e.kind {
                ExternKind::Func => match self.func_type(e.index) {
                    Some(t) => format!("func {}: {t}", e.name),
                    None => format!("func {}: <type index out of range>", e.name),
                },
                ExternKind::Table => format!("table {}", e.name),
                ExternKind::Memory => format!("memory {}", e.name),
                ExternKind::Global => format!("global {}", e.name),
            })
            .collect()
    }
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
    /// An import or export description's kind byte was none of `0x00`
    /// (func), `0x01` (table), `0x02` (memory), `0x03` (global).
    InvalidExternKind(u8),
    /// A table or memory limits' flag byte was neither `0x00` (min only) nor
    /// `0x01` (min and max).
    InvalidLimits,
    /// A custom section's name was not valid UTF-8.
    InvalidUtf8,
    /// A type index named by an import or the function section named a type
    /// that does not exist.
    TypeIndexOutOfRange(u32),
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
            ParseErrorKind::InvalidExternKind(byte) => {
                write!(f, "invalid import/export kind byte {byte:#04x}")
            }
            ParseErrorKind::InvalidLimits => {
                f.write_str("invalid limits flag byte (expected 0x00 or 0x01)")
            }
            ParseErrorKind::InvalidUtf8 => f.write_str("custom section name is not valid UTF-8"),
            ParseErrorKind::TypeIndexOutOfRange(idx) => write!(f, "type index {idx} is out of range"),
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
/// times. Type, import, function, export and start sections are decoded;
/// every other known id is skipped by length and recorded in
/// `Module::skipped_sections`.
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
                match id {
                    1 => module.types = parse_type_section(body, body_start)?,
                    2 => module.imports = parse_import_section(body, body_start, &module.types)?,
                    3 => module.funcs = parse_function_section(body, body_start, &module.types)?,
                    7 => module.exports = parse_export_section(body, body_start)?,
                    8 => module.start = Some(parse_start_section(body, body_start)?),
                    _ => module.skipped_sections.push(id),
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

/// Reads a single byte at `*pos`, advancing past it, or reports the absolute
/// offset (`base + *pos`) as `UnexpectedEof`.
fn read_byte(bytes: &[u8], pos: &mut usize, base: usize) -> Result<u8, ParseError> {
    let offset = base + *pos;
    let byte = *bytes.get(*pos).ok_or(ParseError { offset, kind: ParseErrorKind::UnexpectedEof })?;
    *pos += 1;
    Ok(byte)
}

/// Reads a `limits`: a flag byte (`0x00` min-only, `0x01` min and max)
/// followed by one or two `u32`s. Used by table and memory import types;
/// the values themselves are not kept since this crate gives guest code
/// neither tables nor memory.
fn read_limits(bytes: &[u8], pos: &mut usize, base: usize) -> Result<(), ParseError> {
    let flag_offset = base + *pos;
    let flag = read_byte(bytes, pos, base)?;
    read_u32_at(bytes, pos, base)?; // min
    match flag {
        0x00 => Ok(()),
        0x01 => {
            read_u32_at(bytes, pos, base)?; // max
            Ok(())
        }
        _ => Err(ParseError { offset: flag_offset, kind: ParseErrorKind::InvalidLimits }),
    }
}

/// Decodes the type section: `vec(functype)`, where each `functype` is
/// `0x60 vec(valtype) vec(valtype)` (parameters, then results).
fn parse_type_section(body: &[u8], base: usize) -> Result<Vec<FuncType>, ParseError> {
    let mut pos = 0;
    let count = read_u32_at(body, &mut pos, base)?;
    let mut types = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let tag_offset = base + pos;
        let tag = read_byte(body, &mut pos, base)?;
        if tag != FUNC_TYPE_TAG {
            return Err(ParseError { offset: tag_offset, kind: ParseErrorKind::InvalidFuncType(tag) });
        }
        let params = read_val_type_vec(body, &mut pos, base)?;
        let results = read_val_type_vec(body, &mut pos, base)?;
        types.push(FuncType { params, results });
    }

    // A fully understood section can catch a size field that lied: if the
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
        let byte = read_byte(body, pos, base)?;
        out.push(ValType::from_byte(byte).ok_or(ParseError { offset, kind: ParseErrorKind::InvalidValType(byte) })?);
    }
    Ok(out)
}

/// Decodes the import section: `vec(import)`, where each `import` is
/// `mod:name name:name desc:importdesc`. `importdesc` starts with a kind byte
/// (func/table/memory/global) that determines what follows: a type index for
/// a function, a table type or limits for a table or memory, a value type
/// and mutability flag for a global. Only the function case keeps its
/// payload - `types` is passed in so a function import's type index can be
/// checked against it immediately, the same way the function section checks
/// its own type indices.
fn parse_import_section(body: &[u8], base: usize, types: &[FuncType]) -> Result<Vec<Import>, ParseError> {
    let mut pos = 0;
    let count = read_u32_at(body, &mut pos, base)?;
    let mut imports = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let module = read_name(body, &mut pos, base)?;
        let name = read_name(body, &mut pos, base)?;
        let kind_offset = base + pos;
        let kind_byte = read_byte(body, &mut pos, base)?;
        let desc = match kind_byte {
            0x00 => {
                let idx_offset = base + pos;
                let type_idx = read_u32_at(body, &mut pos, base)?;
                if type_idx as usize >= types.len() {
                    return Err(ParseError { offset: idx_offset, kind: ParseErrorKind::TypeIndexOutOfRange(type_idx) });
                }
                ImportDesc::Func(type_idx)
            }
            0x01 => {
                // Table type: a reftype byte (funcref or externref) that this
                // crate never inspects, since it never gives guest code a
                // table either way, then limits.
                read_byte(body, &mut pos, base)?;
                read_limits(body, &mut pos, base)?;
                ImportDesc::Table
            }
            0x02 => {
                read_limits(body, &mut pos, base)?;
                ImportDesc::Memory
            }
            0x03 => {
                let valtype_offset = base + pos;
                let valtype_byte = read_byte(body, &mut pos, base)?;
                ValType::from_byte(valtype_byte)
                    .ok_or(ParseError { offset: valtype_offset, kind: ParseErrorKind::InvalidValType(valtype_byte) })?;
                read_byte(body, &mut pos, base)?; // mutability flag, not kept
                ImportDesc::Global
            }
            other => return Err(ParseError { offset: kind_offset, kind: ParseErrorKind::InvalidExternKind(other) }),
        };
        imports.push(Import { module, name, desc });
    }

    if pos != body.len() {
        return Err(ParseError { offset: base + pos, kind: ParseErrorKind::SectionSizeMismatch });
    }

    Ok(imports)
}

/// Decodes the function section: `vec(typeidx)`, one entry per locally
/// defined function, in the order the code section's bodies will match up
/// with.
fn parse_function_section(body: &[u8], base: usize, types: &[FuncType]) -> Result<Vec<Func>, ParseError> {
    let mut pos = 0;
    let count = read_u32_at(body, &mut pos, base)?;
    let mut funcs = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let idx_offset = base + pos;
        let type_idx = read_u32_at(body, &mut pos, base)?;
        if type_idx as usize >= types.len() {
            return Err(ParseError { offset: idx_offset, kind: ParseErrorKind::TypeIndexOutOfRange(type_idx) });
        }
        funcs.push(Func { type_idx });
    }

    if pos != body.len() {
        return Err(ParseError { offset: base + pos, kind: ParseErrorKind::SectionSizeMismatch });
    }

    Ok(funcs)
}

/// Decodes the export section: `vec(export)`, where each `export` is
/// `name:name desc:exportdesc` and `exportdesc` is a kind byte followed by an
/// index into that kind's index space.
fn parse_export_section(body: &[u8], base: usize) -> Result<Vec<Export>, ParseError> {
    let mut pos = 0;
    let count = read_u32_at(body, &mut pos, base)?;
    let mut exports = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let name = read_name(body, &mut pos, base)?;
        let kind_offset = base + pos;
        let kind_byte = read_byte(body, &mut pos, base)?;
        let kind = ExternKind::from_byte(kind_byte)
            .ok_or(ParseError { offset: kind_offset, kind: ParseErrorKind::InvalidExternKind(kind_byte) })?;
        let index = read_u32_at(body, &mut pos, base)?;
        exports.push(Export { name, kind, index });
    }

    if pos != body.len() {
        return Err(ParseError { offset: base + pos, kind: ParseErrorKind::SectionSizeMismatch });
    }

    Ok(exports)
}

/// Decodes the start section: a single `funcidx`, with no length prefix of
/// its own beyond the section's.
fn parse_start_section(body: &[u8], base: usize) -> Result<u32, ParseError> {
    let mut pos = 0;
    let index = read_u32_at(body, &mut pos, base)?;
    if pos != body.len() {
        return Err(ParseError { offset: base + pos, kind: ParseErrorKind::SectionSizeMismatch });
    }
    Ok(index)
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

    #[test]
    fn parses_a_function_import() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x01, 0x04, 0x01, 0x60, 0x00, 0x00]); // type 0: () -> ()
        // import section: id 2, one import "env"."log" of type 0
        bytes.extend_from_slice(&[
            0x02, 0x0B, 0x01, 0x03, b'e', b'n', b'v', 0x03, b'l', b'o', b'g', 0x00, 0x00,
        ]);
        let module = parse(&bytes).unwrap();
        assert_eq!(
            module.imports,
            vec![Import {
                module: "env".to_string(),
                name: "log".to_string(),
                desc: ImportDesc::Func(0),
            }]
        );
        assert_eq!(module.imported_func_count(), 1);
    }

    #[test]
    fn parses_table_memory_and_global_imports_and_keeps_reading_after_them() {
        let mut bytes = header();
        bytes.extend_from_slice(&[
            0x02, 0x18, 0x03, // import section, size 24, 3 imports
            0x01, b'a', 0x01, b't', 0x01, 0x70, 0x00, 0x01, // "a"."t": table(funcref, limits{min:1})
            0x01, b'a', 0x01, b'm', 0x02, 0x01, 0x01, 0x02, // "a"."m": memory(limits{min:1,max:2})
            0x01, b'a', 0x01, b'g', 0x03, 0x7F, 0x01, // "a"."g": global(i32, mutable)
        ]);
        let module = parse(&bytes).unwrap();
        assert_eq!(
            module.imports,
            vec![
                Import { module: "a".to_string(), name: "t".to_string(), desc: ImportDesc::Table },
                Import { module: "a".to_string(), name: "m".to_string(), desc: ImportDesc::Memory },
                Import { module: "a".to_string(), name: "g".to_string(), desc: ImportDesc::Global },
            ]
        );
        assert_eq!(module.imported_func_count(), 0);
    }

    #[test]
    fn rejects_a_function_import_with_an_out_of_range_type_index() {
        let mut bytes = header();
        // no type section, so type index 0 does not exist
        bytes.extend_from_slice(&[0x02, 0x07, 0x01, 0x01, b'a', 0x01, b'b', 0x00, 0x00]);
        let idx_offset = bytes.len() - 1;
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: idx_offset, kind: ParseErrorKind::TypeIndexOutOfRange(0) })
        );
    }

    #[test]
    fn rejects_an_invalid_import_kind() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x02, 0x06, 0x01, 0x01, b'a', 0x01, b'b', 0x04]); // kind 4 does not exist
        let kind_offset = bytes.len() - 1;
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: kind_offset, kind: ParseErrorKind::InvalidExternKind(4) })
        );
    }

    #[test]
    fn rejects_invalid_limits_flag() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x02, 0x07, 0x01, 0x01, b'a', 0x01, b'b', 0x02, 0x02]); // flag 2 is neither 0 nor 1
        let flag_offset = bytes.len() - 1;
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: flag_offset, kind: ParseErrorKind::InvalidLimits })
        );
    }

    #[test]
    fn parses_a_function_section_and_checks_its_type_indices() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x01, 0x04, 0x01, 0x60, 0x00, 0x00]); // type 0: () -> ()
        bytes.extend_from_slice(&[0x03, 0x03, 0x02, 0x00, 0x00]); // function section: two funcs, both type 0
        let module = parse(&bytes).unwrap();
        assert_eq!(module.funcs, vec![Func { type_idx: 0 }, Func { type_idx: 0 }]);
    }

    #[test]
    fn rejects_a_function_section_entry_with_an_out_of_range_type_index() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x01, 0x04, 0x01, 0x60, 0x00, 0x00]); // type 0
        bytes.extend_from_slice(&[0x03, 0x02, 0x01, 0x01]); // one func, type 1 (does not exist)
        let idx_offset = bytes.len() - 1;
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: idx_offset, kind: ParseErrorKind::TypeIndexOutOfRange(1) })
        );
    }

    #[test]
    fn parses_exports_and_resolves_function_types_through_them() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x01, 0x06, 0x01, 0x60, 0x01, 0x7F, 0x01, 0x7F]); // type 0: (i32) -> i32
        bytes.extend_from_slice(&[0x03, 0x02, 0x01, 0x00]); // one func, type 0
        bytes.extend_from_slice(&[0x07, 0x0A, 0x01, 0x06, b's', b'q', b'u', b'a', b'r', b'e', 0x00, 0x00]); // export "square" func 0
        let module = parse(&bytes).unwrap();
        assert_eq!(
            module.exports,
            vec![Export { name: "square".to_string(), kind: ExternKind::Func, index: 0 }]
        );
        assert_eq!(module.export_func("square"), Some(0));
        assert_eq!(module.export_func("missing"), None);
        assert_eq!(
            module.func_type(0),
            Some(&FuncType { params: vec![ValType::I32], results: vec![ValType::I32] })
        );
        assert_eq!(module.describe_exports(), vec!["func square: (i32) -> i32".to_string()]);
    }

    #[test]
    fn rejects_an_invalid_export_kind() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x07, 0x05, 0x01, 0x01, b'x', 0x04, 0x00]); // kind 4 does not exist
        let kind_offset = bytes.len() - 2;
        assert_eq!(
            parse(&bytes),
            Err(ParseError { offset: kind_offset, kind: ParseErrorKind::InvalidExternKind(4) })
        );
    }

    #[test]
    fn parses_a_start_section() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x01, 0x04, 0x01, 0x60, 0x00, 0x00]); // type 0
        bytes.extend_from_slice(&[0x03, 0x02, 0x01, 0x00]); // one func, type 0
        bytes.extend_from_slice(&[0x08, 0x01, 0x00]); // start: func 0
        let module = parse(&bytes).unwrap();
        assert_eq!(module.start, Some(0));
    }

    #[test]
    fn imported_functions_occupy_the_low_end_of_the_function_index_space() {
        let mut bytes = header();
        bytes.extend_from_slice(&[0x01, 0x04, 0x01, 0x60, 0x00, 0x00]); // type 0: () -> ()
        bytes.extend_from_slice(&[
            0x02, 0x0B, 0x01, 0x03, b'e', b'n', b'v', 0x03, b'l', b'o', b'g', 0x00, 0x00,
        ]); // import 0: func, type 0
        bytes.extend_from_slice(&[0x03, 0x02, 0x01, 0x00]); // local func at index 1, type 0
        let module = parse(&bytes).unwrap();
        assert_eq!(module.imported_func_count(), 1);
        assert_eq!(module.func_type(0), Some(&FuncType { params: vec![], results: vec![] }));
        assert_eq!(module.func_type(1), Some(&FuncType { params: vec![], results: vec![] }));
        assert_eq!(module.func_type(2), None);
    }
}

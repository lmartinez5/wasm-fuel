//! The WebAssembly binary format decoder.
//!
//! This module builds up the module structure one piece at a time: header
//! first, then sections. What is here so far is just the header - every
//! binary starts with a 4-byte magic number and a 4-byte version, and both
//! have to be checked before anything else is worth reading.

use std::fmt;

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
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ParseErrorKind::NotWasm => "not a WebAssembly binary (bad magic number)",
            ParseErrorKind::UnsupportedVersion => "unsupported WebAssembly version",
            ParseErrorKind::UnexpectedEof => "unexpected end of input",
        })
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
}

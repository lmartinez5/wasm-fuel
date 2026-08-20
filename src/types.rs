//! The value and type vocabulary shared by the parser and the interpreter:
//! `ValType` (what the binary format encodes), `FuncType` (a signature), and
//! `Val` (a runtime value tagged with its type).

use std::fmt;

/// A WebAssembly value type.
///
/// The binary format also defines reference types (`funcref`, `externref`)
/// and, in the SIMD proposal, `v128`. Nothing in this crate produces or
/// consumes those, so they have no variant here; the parser rejects their
/// byte encodings with `ParseError::InvalidValType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

impl ValType {
    /// Decodes the single byte the binary format uses for a value type,
    /// e.g. in a function signature or a local declaration.
    pub fn from_byte(byte: u8) -> Option<ValType> {
        match byte {
            0x7F => Some(ValType::I32),
            0x7E => Some(ValType::I64),
            0x7D => Some(ValType::F32),
            0x7C => Some(ValType::F64),
            _ => None,
        }
    }
}

impl fmt::Display for ValType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ValType::I32 => "i32",
            ValType::I64 => "i64",
            ValType::F32 => "f32",
            ValType::F64 => "f64",
        })
    }
}

/// A function signature: an ordered list of parameter types followed by an
/// ordered list of result types.
///
/// WebAssembly 1.0 allows at most one result, but the field is a `Vec`
/// rather than an `Option` so that a module using the multi-value proposal
/// still decodes into something this type can represent; only the parser and
/// interpreter would need to grow to actually accept more than one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

impl fmt::Display for FuncType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{p}")?;
        }
        f.write_str(") -> ")?;
        match self.results.as_slice() {
            [] => f.write_str("()"),
            [only] => write!(f, "{only}"),
            many => {
                f.write_str("(")?;
                for (i, r) in many.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{r}")?;
                }
                f.write_str(")")
            }
        }
    }
}

/// A runtime value: an operand on the interpreter's value stack, a local, or
/// an argument or result of a call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Val {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl Val {
    /// The type of this value, for checking against a `FuncType` or a local
    /// declaration.
    pub fn val_type(&self) -> ValType {
        match self {
            Val::I32(_) => ValType::I32,
            Val::I64(_) => ValType::I64,
            Val::F32(_) => ValType::F32,
            Val::F64(_) => ValType::F64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_val_type_bytes() {
        assert_eq!(ValType::from_byte(0x7F), Some(ValType::I32));
        assert_eq!(ValType::from_byte(0x7E), Some(ValType::I64));
        assert_eq!(ValType::from_byte(0x7D), Some(ValType::F32));
        assert_eq!(ValType::from_byte(0x7C), Some(ValType::F64));
        assert_eq!(ValType::from_byte(0x70), None); // funcref, not accepted
        assert_eq!(ValType::from_byte(0x00), None);
    }

    #[test]
    fn val_reports_its_own_type() {
        assert_eq!(Val::I32(9).val_type(), ValType::I32);
        assert_eq!(Val::I64(-1).val_type(), ValType::I64);
        assert_eq!(Val::F32(1.5).val_type(), ValType::F32);
        assert_eq!(Val::F64(1.5).val_type(), ValType::F64);
    }

    #[test]
    fn displays_signatures_like_the_readme_examples() {
        let square = FuncType { params: vec![ValType::I32], results: vec![ValType::I32] };
        assert_eq!(square.to_string(), "(i32) -> i32");

        let no_args_no_result = FuncType { params: vec![], results: vec![] };
        assert_eq!(no_args_no_result.to_string(), "() -> ()");

        let two_params = FuncType { params: vec![ValType::I32, ValType::I64], results: vec![ValType::F64] };
        assert_eq!(two_params.to_string(), "(i32, i64) -> f64");

        let multi_result = FuncType { params: vec![], results: vec![ValType::I32, ValType::I32] };
        assert_eq!(multi_result.to_string(), "() -> (i32, i32)");
    }
}

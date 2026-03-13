//! Attribute values attached to IR operations.
//!
//! Attributes carry static, compile-time configuration for an operation (e.g.
//! kernel size, strides, padding mode).  They are distinct from tensor operands:
//! attributes are always known at graph-construction time.

use std::fmt;

use serde::Serialize;

/// Padding mode for convolution and pooling operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum PaddingMode {
    /// No padding — output spatial dimensions shrink.
    Valid,
    /// Pad so that output spatial dimensions equal `ceil(input / stride)`.
    Same,
    /// Explicit per-side padding provided via [`Attr::Pads`].
    Explicit,
}

impl fmt::Display for PaddingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid => f.write_str("valid"),
            Self::Same => f.write_str("same"),
            Self::Explicit => f.write_str("explicit"),
        }
    }
}

/// An attribute value that parameterises an IR operation.
///
/// This enum is intentionally kept small and will grow as new operations
/// require additional configuration knobs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Attr {
    /// A single integer (e.g. axis index, group count).
    Int(i64),
    /// A list of integers (e.g. `kernel_size`, strides, dilations).
    Ints(Vec<i64>),
    /// A single floating-point value.
    Float(f64),
    /// A list of floats.
    Floats(Vec<f64>),
    /// A string value (e.g. activation function name).
    String(String),
    /// Padding mode selector.
    Padding(PaddingMode),
}

impl fmt::Display for Attr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Ints(vs) => {
                write!(f, "[")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Self::Float(v) => write!(f, "{v}"),
            Self::Floats(vs) => {
                write!(f, "[")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Padding(p) => write!(f, "{p}"),
        }
    }
}

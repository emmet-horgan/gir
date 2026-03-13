//! Data types supported by the IR, focused on embedded ML workloads.
//!
//! The type system is intentionally small — it captures the quantised integer
//! and reduced-precision floating-point formats that dominate embedded inference,
//! while remaining open for extension (e.g. bfloat16, fp8).

use std::fmt;

use serde::Serialize;

/// Scalar element data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum DType {
    /// Boolean (1-bit logical).
    Bool,
    /// Signed 8-bit integer (common quantisation target).
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 32-bit integer (accumulators, biases).
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// IEEE 754 half-precision (16-bit) float.
    F16,
    /// IEEE 754 single-precision (32-bit) float.
    F32,
}

impl DType {
    /// Size of one element in bits.
    #[must_use]
    pub const fn bit_width(self) -> u32 {
        match self {
            Self::Bool => 1,
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
        }
    }

    /// Size of one element in bytes (rounded up for sub-byte types).
    #[must_use]
    pub const fn byte_width(self) -> u32 {
        self.bit_width().div_ceil(8)
    }

    /// Whether the type is an integer (signed or unsigned).
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Bool | Self::I8 | Self::U8 | Self::I16 | Self::U16 | Self::I32 | Self::U32
        )
    }

    /// Whether the type is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32)
    }

    /// Whether the type is signed.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::F16 | Self::F32
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::F16 => "f16",
            Self::F32 => "f32",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_widths() {
        assert_eq!(DType::Bool.bit_width(), 1);
        assert_eq!(DType::I8.bit_width(), 8);
        assert_eq!(DType::F16.bit_width(), 16);
        assert_eq!(DType::F32.bit_width(), 32);
    }

    #[test]
    fn byte_widths() {
        assert_eq!(DType::Bool.byte_width(), 1); // rounded up
        assert_eq!(DType::I8.byte_width(), 1);
        assert_eq!(DType::F32.byte_width(), 4);
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", DType::I8), "i8");
        assert_eq!(format!("{}", DType::F32), "f32");
    }
}

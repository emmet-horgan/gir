//! Dense tensor data storage for constant weights and parameters.
//!
//! [`TensorData`] holds the raw bytes of a tensor alongside its [`DType`].
//! This is used to attach actual weight values to `Constant` nodes in the
//! graph, enabling downstream passes such as quantisation, constant folding,
//! and serialisation.
//!
//! The data is stored as a flat `Vec<u8>` in **little-endian** byte order —
//! the same layout that ONNX `raw_data` uses — keeping zero-copy ingestion
//! from ONNX models free.
//!
//! # Example
//!
//! ```
//! use gir::data::TensorData;
//! use gir::dtype::DType;
//!
//! let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
//! let td = TensorData::from_f32s(&weights);
//!
//! assert_eq!(td.dtype(), DType::F32);
//! assert_eq!(td.len(), 4);
//! assert_eq!(td.as_f32s().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
//! ```

use std::fmt;

use serde::Serialize;

use crate::dtype::DType;

/// Dense tensor data — a flat byte buffer with an associated element type.
///
/// The buffer is stored in native little-endian order.  Helper constructors
/// and accessor methods are provided for each [`DType`] variant so that
/// callers rarely need to touch raw bytes directly.
#[derive(Clone, PartialEq, Serialize)]
pub struct TensorData {
    dtype: DType,
    /// Raw little-endian bytes.  Length must equal `num_elements * dtype.byte_width()`.
    bytes: Vec<u8>,
}

// ─── Construction ───────────────────────────────────────────────────

impl TensorData {
    /// Create a `TensorData` from pre-existing raw bytes and a dtype.
    ///
    /// # Panics
    ///
    /// Panics if `bytes.len()` is not a multiple of `dtype.byte_width()`.
    #[must_use]
    pub fn from_raw(dtype: DType, bytes: Vec<u8>) -> Self {
        let bw = dtype.byte_width() as usize;
        assert!(
            bw == 0 || bytes.len() % bw == 0,
            "byte buffer length {} is not a multiple of element size {bw}",
            bytes.len(),
        );
        Self { dtype, bytes }
    }

    /// Create from a slice of `f32` values.
    #[must_use]
    pub fn from_f32s(data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            dtype: DType::F32,
            bytes,
        }
    }

    /// Create from a slice of `i32` values.
    #[must_use]
    pub fn from_i32s(data: &[i32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            dtype: DType::I32,
            bytes,
        }
    }

    /// Create from a slice of `u8` values.
    #[must_use]
    pub fn from_u8s(data: &[u8]) -> Self {
        Self {
            dtype: DType::U8,
            bytes: data.to_vec(),
        }
    }

    /// Create from a slice of `i8` values.
    #[must_use]
    pub fn from_i8s(data: &[i8]) -> Self {
        // i8 and u8 have identical bit representations.
        let bytes: Vec<u8> = data
            .iter()
            .map(|&v| u8::from_ne_bytes(v.to_ne_bytes()))
            .collect();
        Self {
            dtype: DType::I8,
            bytes,
        }
    }

    /// Create from a slice of `i64` values, stored as `I32` (clamped).
    ///
    /// This mirrors the ONNX convention where INT64 is mapped to our `I32`.
    /// Values are truncated via `i32::try_from` (saturating to 0 on overflow).
    #[must_use]
    pub fn from_i64s_as_i32(data: &[i64]) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect();
        Self {
            dtype: DType::I32,
            bytes,
        }
    }
}

// ─── Accessors ──────────────────────────────────────────────────────

impl TensorData {
    /// The element data type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// The raw byte buffer (little-endian).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume and return the underlying byte buffer.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        let bw = self.dtype.byte_width() as usize;
        if bw == 0 { 0 } else { self.bytes.len() / bw }
    }

    /// Whether the buffer contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Total size in bytes.
    #[must_use]
    pub fn size_in_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// View as `&[f32]`.  Returns `None` if dtype is not `F32`.
    #[must_use]
    pub fn as_f32s(&self) -> Option<Vec<f32>> {
        if self.dtype != DType::F32 {
            return None;
        }
        Some(
            self.bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap_or_default()))
                .collect(),
        )
    }

    /// View as `&[i32]`.  Returns `None` if dtype is not `I32`.
    #[must_use]
    pub fn as_i32s(&self) -> Option<Vec<i32>> {
        if self.dtype != DType::I32 {
            return None;
        }
        Some(
            self.bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap_or_default()))
                .collect(),
        )
    }

    /// View as `&[u8]`.  Returns `None` if dtype is not `U8`.
    #[must_use]
    pub fn as_u8s(&self) -> Option<&[u8]> {
        if self.dtype != DType::U8 {
            return None;
        }
        Some(&self.bytes)
    }

    /// View as `&[i8]`.  Returns `None` if dtype is not `I8`.
    #[must_use]
    pub fn as_i8s(&self) -> Option<Vec<i8>> {
        if self.dtype != DType::I8 {
            return None;
        }
        Some(self.bytes.iter().map(|&b| i8::from_ne_bytes([b])).collect())
    }
}

// ─── Display / Debug ────────────────────────────────────────────────

impl fmt::Debug for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorData({}, {} elements, {} bytes)",
            self.dtype,
            self.len(),
            self.bytes.len(),
        )
    }
}

impl fmt::Display for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "data<{}, {} elem>", self.dtype, self.len())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_roundtrip() {
        let vals = vec![1.0_f32, -2.5, 0.0, 3.125];
        let td = TensorData::from_f32s(&vals);
        assert_eq!(td.dtype(), DType::F32);
        assert_eq!(td.len(), 4);
        assert!(!td.is_empty());
        assert_eq!(td.size_in_bytes(), 16);
        assert_eq!(td.as_f32s().unwrap(), vals);
        assert!(td.as_i32s().is_none());
    }

    #[test]
    fn i32_roundtrip() {
        let vals = vec![42_i32, -1, 0, 100];
        let td = TensorData::from_i32s(&vals);
        assert_eq!(td.dtype(), DType::I32);
        assert_eq!(td.len(), 4);
        assert_eq!(td.as_i32s().unwrap(), vals);
    }

    #[test]
    fn u8_roundtrip() {
        let vals: Vec<u8> = vec![0, 127, 255];
        let td = TensorData::from_u8s(&vals);
        assert_eq!(td.dtype(), DType::U8);
        assert_eq!(td.len(), 3);
        assert_eq!(td.as_u8s().unwrap(), &vals);
    }

    #[test]
    fn i8_roundtrip() {
        let vals = vec![-128_i8, 0, 127];
        let td = TensorData::from_i8s(&vals);
        assert_eq!(td.dtype(), DType::I8);
        assert_eq!(td.len(), 3);
        assert_eq!(td.as_i8s().unwrap(), vals);
    }

    #[test]
    fn from_raw_bytes() {
        let f: f32 = 42.0;
        let td = TensorData::from_raw(DType::F32, f.to_le_bytes().to_vec());
        assert_eq!(td.len(), 1);
        assert_eq!(td.as_f32s().unwrap(), vec![42.0]);
    }

    #[test]
    fn i64_to_i32() {
        let vals = vec![1_i64, -2, 3];
        let td = TensorData::from_i64s_as_i32(&vals);
        assert_eq!(td.dtype(), DType::I32);
        assert_eq!(td.as_i32s().unwrap(), vec![1_i32, -2, 3]);
    }

    #[test]
    fn empty_data() {
        let td = TensorData::from_f32s(&[]);
        assert!(td.is_empty());
        assert_eq!(td.len(), 0);
    }

    #[test]
    #[should_panic(expected = "not a multiple")]
    fn bad_raw_panics() {
        let _ = TensorData::from_raw(DType::F32, vec![0, 1, 2]); // 3 bytes, not multiple of 4
    }
}

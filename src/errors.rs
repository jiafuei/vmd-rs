use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VmdError {
    #[error("shape error {0}")]
    InputShapeError(ndarray::ShapeError),
}

impl From<ndarray::ShapeError> for VmdError {
    fn from(value: ndarray::ShapeError) -> Self {
        VmdError::InputShapeError(value)
    }
}

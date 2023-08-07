use ndarray::{s, ArrayBase, ArrayView1, Axis, Data, Ix1, OwnedRepr};

pub trait Flip<A, S> {
    fn flip(&self) -> ArrayView1<A>;
}

impl<A, S> Flip<A, S> for ArrayBase<S, Ix1>
where
    A: Clone,
    S: Data<Elem = A>,
{
    fn flip(&self) -> ArrayView1<A>
    where
        A: Clone,
        S: Data<Elem = A>,
    {
        self.slice(s![0..; -1])
    }
}

pub fn fftshift1d<A, S>(arr: ArrayBase<S, Ix1>) -> ArrayBase<OwnedRepr<A>, Ix1>
where
    A: Clone,
    S: Data<Elem = A>,
{
    let half = if arr.len() % 2 == 0 {
        arr.len() / 2
    } else {
        (arr.len() / 2) + 1
    };
    let first_half = arr.slice(s![..half]);
    let second_half = arr.slice(s![half..]);
    ndarray::concatenate![Axis(0), second_half, first_half]
}

pub fn ifftshift1d<A, S>(arr: ArrayBase<S, Ix1>) -> ArrayBase<OwnedRepr<A>, Ix1>
where
    A: Clone,
    S: Data<Elem = A>,
{
    let even = arr.len() % 2 == 0;
    if even {
        return fftshift1d(arr);
    }
    let half = arr.len() / 2;
    let first_half = arr.slice(s![..half]);
    let second_half = arr.slice(s![half..]);
    ndarray::concatenate![Axis(0), second_half, first_half]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, ArrayView1};

    #[test]
    fn test_flip_ok() {
        let input = [1, 2, 3, 4, 5];
        let expected = [5, 4, 3, 2, 1];
        let input_view = ndarray::ArrayView1::from_shape(input.len(), &input).unwrap();
        let output = input_view.flip();
        let expected = ndarray::ArrayView1::from_shape(expected.len(), &expected).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_fftshift_even() {
        let input = [1, 2, 3, 4, 5, 6];
        let expected = [4, 5, 6, 1, 2, 3];
        let expected = ArrayView1::from_shape(expected.len(), &expected).unwrap();
        let input = Array1::from_iter(input);

        let output = fftshift1d(input);
        assert_eq!(&output, expected);
    }

    /// Match python's behaviour
    #[test]
    fn test_fftshift_odd() {
        let input = [1, 2, 3, 4, 5, 6, 7];
        let expected = [5, 6, 7, 1, 2, 3, 4];
        let expected = ArrayView1::from_shape(expected.len(), &expected).unwrap();
        let input = Array1::from_iter(input);

        let output = fftshift1d(input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_ifftshift_even() {
        let input = [4, 5, 6, 1, 2, 3];
        let expected = [1, 2, 3, 4, 5, 6];
        let expected = ArrayView1::from_shape(expected.len(), &expected).unwrap();
        let input = Array1::from_iter(input);

        let output = ifftshift1d(input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_ifftshift_odd() {
        let input = [5, 6, 7, 1, 2, 3, 4];
        let expected = [1, 2, 3, 4, 5, 6, 7];
        let expected = ArrayView1::from_shape(expected.len(), &expected).unwrap();
        let input = Array1::from_iter(input);

        let output = ifftshift1d(input);
        assert_eq!(output, expected);

        let input = [6, 7, 8, 9, 1, 2, 3, 4, 5];
        let expected = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let expected = ArrayView1::from_shape(expected.len(), &expected).unwrap();
        let input = Array1::from_iter(input);

        let output = ifftshift1d(input);
        assert_eq!(output, expected);
    }
}

use approx::{assert_relative_eq, RelativeEq};
use ndarray::Zip;
use num_complex::Complex;
use vmd_rs::vmd;

#[test]
// Test using output from vmdpy
fn test_vmd_ok() {
    let test_input: ndarray::Array1<f64> = ndarray_npy::read_npy("tests/sample/input.npy").unwrap();
    let expected_ulong: ndarray::Array2<f64> =
        ndarray_npy::read_npy("tests/sample/ulong.npy").unwrap();
    let expected_uhat: ndarray::Array2<Complex<f64>> =
        ndarray_npy::read_npy("tests/sample/uhat.npy").unwrap();
    let expected_omega: ndarray::Array2<f64> =
        ndarray_npy::read_npy("tests/sample/omega.npy").unwrap();

    let (u, u_hat, omega) = vmd(test_input.as_slice().unwrap(), 2000., 0., 4, 0, 1, 1e-7).unwrap();

    assert_relative_eq!(u, expected_ulong, max_relative = 0.0000000005);
    assert_relative_eq!(omega, expected_omega, max_relative = 0.0000000005);
    let uhat_eq = Zip::from(u_hat.rows())
        .and(expected_uhat.rows())
        .all(|a, b| {
            Zip::from(a).and(b).all(|&x, y| {
                x.re.relative_eq(&y.re, f64::EPSILON, 0.0000005)
                    && x.im.relative_eq(&y.im, f64::EPSILON, 0.0000005)
            })
        });
    assert!(uhat_eq);
}

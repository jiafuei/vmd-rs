use ndarray::{concatenate, prelude::*};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ndarray_slice::Slice1Ext;
use num_complex::{Complex, ComplexFloat};
use rustfft::FftPlanner;
use std::cell::RefCell;

use crate::errors::VmdError;
use crate::utils::array::{fftshift1d, ifftshift1d, Flip};

thread_local! {
  static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

#[allow(non_snake_case, clippy::type_complexity)]
/// Description
/// ---
/// u,u_hat,omega = VMD(input, alpha, tau, K, DC, init, tol) <br>
/// Variational mode decomposition <br>
/// Based on Python implementation by @vrcarfa <br>
/// Original paper: <br>
/// Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’,  <br>
/// IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675. <br>
///
/// Input and Parameters:
/// ---------------------
/// input   - the time domain signal (1D) to be decomposed <br>
/// alpha   - the balancing parameter of the data-fidelity constraint <br>
/// tau     - time-step of the dual ascent ( pick 0 for noise-slack ) <br>
/// K       - the number of modes to be recovered <br>
/// DC      - true if the first mode is put and kept at DC (0-freq) <br>
/// init    - 0 = all omegas start at 0 <br>
///           1 = all omegas start uniformly distributed <br>
///           2 = all omegas initialized randomly <br>
/// tol     - tolerance of convergence criterion; typically around 1e-6 <br>
///
/// Output:
/// -------
/// u       - the collection of decomposed modes <br>
/// u_hat   - spectra of the modes <br>
/// omega   - estimated mode center-frequencies <br>
///
pub fn vmd(
    input: &[f64],
    alpha: i32,
    tau: i32,
    K: usize,
    DC: i32,
    init: i32,
    tol: f64,
) -> Result<(Array2<f64>, Array2<Complex<f64>>, Array2<f64>), VmdError> {
    // Output of python code did not work for odd number

    // Period and sampling of input frequency
    let fs = 1.0 / input.len() as f64;

    let T = input.len();
    let midpoint = (input.len() as f64 / 2.0).ceil() as usize;

    let mut f_mirr = {
        let input = ArrayView1::from_shape(T, input)?;
        let first_half = input.slice(s![..midpoint]);
        let second_half = input.slice(s![midpoint..]);
        concatenate(Axis(0), &[first_half.flip(), input, second_half.flip()])?
            // .unwrap()
            .map(|&f| Complex::new(f, 0.))
    };

    let T = f_mirr.len() as f64;
    let t = Array::range(1., T + 1., 1.) / T;
    let t_len = t.len();
    let freqs = t - 0.5 - (1. / T);
    const N_ITER: usize = 500;

    // Construct and center f_hat
    let fft_fhat = {
        FFT_PLANNER.with(|planner| {
            let fft = planner.borrow_mut().plan_fft_forward(T as usize);
            fft.process(f_mirr.as_slice_mut().unwrap());
            f_mirr
            // # Safety
            // The output buffer is immediately filled in the .process() call below
            // let mut output_buf = unsafe {
            // Real-to-Complex FFT skips redundant calculations, effectively returning N/2 + 1 values
            //     // let arr = Array1::uninit((T as usize / 2) + 1);
            //     let arr = Array1::uninit(T as usize);
            //     let arr = arr.assume_init();
            //     arr
            // };
            // return match fft.process(
            //     f_mirr.as_slice_mut().unwrap(),
            //     output_buf.as_slice_mut().unwrap(),
            // ) {
            //     Ok(_) => output_buf,
            //     Err(e) => panic!("{}", e),
            // };
        })
    };

    let f_hat = fftshift1d(fft_fhat.view());
    let mut f_hat_plus = f_hat;
    f_hat_plus
        .slice_mut(s![..T as usize / 2])
        .map_inplace(|v| *v = Complex::new(0., 0.));

    // Initialization of omega k
    let mut omega_plus = Array::from_shape_fn((N_ITER, K), |(_, _)| 0.);
    match init {
        1 => {
            for i in 0..K {
                omega_plus[[0, i]] = (0.5 / K as f64) * i as f64
            }
        }
        // PY => omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
        2 => {
            // TODO: reduce allocs
            let rexpr = fs.log(std::f64::consts::E);
            let random = Array::random([1, K], Uniform::new(0., 1.));
            // let random = ndarray::Array2::from_shape_vec([1, 4], vec![1., 2., 3., 4.]).unwrap();
            let rexpr2 = (0.5_f64.log(std::f64::consts::E) - rexpr) * random;
            let mut expr = rexpr + rexpr2;

            expr.map_inplace(|f| *f = f.exp());
            let mut axis_sort = expr.slice_axis_mut(Axis(0), ndarray::Slice::new(0, None, 1));
            axis_sort
                .row_mut(0)
                .sort_unstable_by(|f1, f2| f1.partial_cmp(f2).unwrap());
            expr.row_mut(0).assign_to(
                omega_plus
                    .slice_axis_mut(Axis(0), ndarray::Slice::new(0, None, 1))
                    .row_mut(0),
            );
        }
        _ => {
            omega_plus.slice_mut(s![.., ..]).map_inplace(|f| *f = 0.);
        }
    };
    if DC != 0 {
        omega_plus[[0, 0]] = 0.;
    }

    // Huge allocs here
    // start with empty dual variables

    // optimization: only need 2 rows, but we use 3 because its simpler to write
    const ROWS: usize = 3;
    let mut lambda_hat: Array2<Complex<f64>> = Array::zeros((ROWS, freqs.len()));

    // Huge allocs!
    // matrix keeping track of every iterant // could be discarded for mem
    // optimization: use only 3 rows
    let mut u_hat_plus: Array3<Complex<f64>> = Array::zeros((ROWS, freqs.len(), K));
    let mut udiff = tol + f64::EPSILON;
    let mut n = 0;
    let mut sum_uk: Array1<Complex<f64>> = Array::zeros(freqs.len());

    let mut cur: usize = 0; // n % ROWS
    let mut next: usize = 1; // (n+1) % ROWS
    let mut prev: usize;

    // For future generalizations: individual alpha for each mode
    let alpha: Array1<f64> = Array::ones(K) * alpha as f64;

    // Main loop for iterative updates
    while udiff > tol && n < N_ITER - 1 {
        let T = T as usize;
        // Not converged and below iteration limit

        // update first mode accumulator
        let k = 0;
        let s1 = u_hat_plus.slice(s![cur, .., K - 1]);
        let s2 = u_hat_plus.slice(s![cur, .., 0]);
        sum_uk += &s1;
        sum_uk -= &s2;

        // Update spectrum of first mode through Wiener filter of residuals
        let lambda_hat_slice = &lambda_hat.slice(s![cur, ..]) / Complex::new(2., 0.);
        let lexpr = &f_hat_plus - &sum_uk - &lambda_hat_slice;
        let rexpr = 1. + alpha[k] * (&freqs - omega_plus[[n, k]]).map_mut(|f| f.powi(2));
        (lexpr / rexpr).move_into(u_hat_plus.slice_mut(s![next, .., k]));

        if DC == 0 {
            let expr1 = freqs.slice(s![T / 2..T]);
            let subexpr2 = u_hat_plus.slice(s![next, T / 2..T, k]);
            let expr2 = subexpr2.map(|f| ComplexFloat::abs(*f).powi(2));
            let expr1: f64 = expr1.dot(&expr2);
            let expr2 = expr2.sum();
            omega_plus[[n + 1, k]] = expr1 / expr2;
        }

        // update of any other node
        for k in 1..K {
            // accumulator
            sum_uk += &u_hat_plus.slice(s![next, .., k - 1]);
            sum_uk -= &u_hat_plus.slice(s![cur, .., k]);

            // mode spectrum
            // let lexpr = &lambda_hat.slice(s![cur, ..]) / Complex::new(2., 0.);
            let lexpr = &f_hat_plus - &sum_uk - &lambda_hat_slice;
            let rexpr = 1. + alpha[k] * (&freqs - omega_plus[[n, k]]).map(|v| v.powi(2));
            (lexpr / rexpr).move_into(u_hat_plus.slice_mut(s![next, .., k]));

            // center frequencies
            let expr1 = freqs.slice(s![T / 2..T]);
            let subexpr2 = u_hat_plus.slice(s![next, T / 2..T, k]);
            let expr2 = subexpr2.map(|f| ComplexFloat::abs(*f).powi(2));
            let expr1: f64 = expr1.dot(&expr2);
            let expr2 = expr2.sum();
            omega_plus[[n + 1, k]] = expr1 / expr2;
        }

        // dual ascent
        let expr1 = (&u_hat_plus
            .slice(s![next, .., ..])
            .sum_axis(ndarray::Axis(1))
            - &f_hat_plus)
            * tau as f64;
        let expr1 = &lambda_hat.slice(s![cur, ..]) + expr1;
        expr1.move_into(lambda_hat.slice_mut(s![next, ..]));

        // loop counters
        n += 1;
        cur = n % ROWS;
        next = (n + 1) % ROWS;
        prev = (n - 1) % ROWS;

        let mut udiff_ = Complex::new(f64::EPSILON, 0.);
        for i in 0..K {
            let expr1 = &u_hat_plus.slice(s![cur, .., i]) - &u_hat_plus.slice(s![prev, .., i]);
            let expr2 = expr1.map(|f| f.conj());
            let expr = expr1.dot(&expr2) * (1. / T as f64);

            udiff_ += expr;
        }
        udiff = ComplexFloat::abs(udiff_);
    }
    // Postprocessing and cleanup
    // discard empty space if converged early
    let n_iter = std::cmp::min(n, N_ITER);
    let omega = omega_plus.slice(s![..n_iter, ..]);

    // signal reconstruction (slight optimization)

    // let idxs = np_flip(&ndarray::Array::range(1., T as f64/2.+1., 1.).view());
    // let idxs = ndarray::Array::range(T as f64/2., 0., -1.);
    // let idxs = T/2..0;
    let T = T as usize;
    let mut u_hat = Array::from_elem([T, K], Complex::new(0.0, 0.0));
    u_hat
        .slice_mut(s![T / 2..T, ..])
        .assign(&u_hat_plus.slice(s![(n_iter - 1) % ROWS, T / 2..T, ..]));
    // idxs = 1..T/2+1;-1
    u_hat_plus
        .slice(s![(n_iter - 1) % ROWS, T / 2..T, ..])
        .map(|f| f.conj())
        .move_into(u_hat.slice_mut(s![1..T/2+1;-1,..]));
    u_hat
        .slice(s![-1, ..])
        .map(|f| f.conj())
        .move_into(u_hat.slice_mut(s![0, ..]));

    let mut u: Array2<f64> = ndarray::Array::zeros([K, t_len]);
    FFT_PLANNER.with(|planner| {
        let ffti = planner
            .borrow_mut()
            .plan_fft_inverse(u_hat.slice(s![.., 0]).len());
        for k in 0..K {
            let subexpr = u_hat.slice(s![.., k]);
            let mut ishifted = ifftshift1d(subexpr);
            ffti.process(ishifted.as_slice_mut().unwrap());
            // rustfft does not normalize, normalize ourselves
            // https://numpy.org/doc/stable/reference/routines.fft.html#normalization
            let len = ishifted.len() as f64;
            // ishifted = ishifted / len;
            // println!("{:?}", &ishifted);
            (ishifted / len)
                .map(|f| f.re())
                .move_into(u.slice_mut(s![k, ..]));
        }
    });

    // Remove mirror part
    let u = u.slice_mut(s![.., T / 4..3 * T / 4]);

    // Recompute spectrum
    let mut u_hat: Array2<Complex<f64>> = Array::zeros([u.shape()[1], K]);
    FFT_PLANNER.with(|planner| {
        for k in 0..K {
            let mut u_ = u.slice(s![k, ..]).map(|f| Complex::new(*f, 0.));
            let fft = planner.borrow_mut().plan_fft_forward(u_.len());
            fft.process(u_.as_slice_mut().unwrap());
            fftshift1d(u_.view()).move_into(u_hat.slice_mut(s![.., k]));
        }
    });

    Ok((u.to_owned(), u_hat, omega.to_owned()))
}

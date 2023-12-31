vmd-rs
---
VMD, aka Variational Mode Decomposition, is a signal processing tool that decompse the input signal into different band-limited IMFs.

The implementation of this project is referenced from [vmdpy](https://github.com/vrcarva/vmdpy) with some slight changes.


Highlights
---

1. Consumes less memory
2. Handles odd number signal length
3. Rust

Using with cargo
---
```
[dependencies]
vmd-rs = "0.2.1"
```

Enabling BLAS
---
Blas integration is optional. See the blas section of [ndarray](https://github.com/rust-ndarray/ndarray) on how to link to blas providers.

Using with Python
---
See [vmdrs-py](https://github.com/jiafuei/vmdrs-py) for an example.

Support
---
Maybe build issues, thats it. I don't understand signal processing. I just translated the Python code to Rust and added some optimizations along the way.

Feel free to make a PR for changes you would like to see.

Credits
---
- [Vinícius R. Carvalho et al.](https://github.com/vrcarva/vmdpy)
- [LordZachery](https://github.com/vrcarva/vmdpy/issues/7#issuecomment-1537228907)

Contribution
---
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
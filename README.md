## Matrix-rs
A (very) small, reasonably ergonomic matrix library.

#### Why?
I didn't like the ergonomics of nalgebra or ndarray.  This was a quick attempt to put together a matrix library that would let me do all the things I wanted with minimal fuss.  It's not as feature complete or fast as either of the others, but I've found it nicer to use and more understandable when I just want to do something fast and quick.  It something akin to a child's drawing of Python's Numpy -- affectionate and well intended, if utterly terrible.

#### Samples
```rust
// Create a new row matrix with values 1, 2, 3.
let m = Matrix::<u32>::from_data(1, 3, vec![1, 2, 3]);

// Create a column matrix with values 1.0, 3.1415, and 1.626
let n = Matrix::<f32>::from_data(3, 1, vec![1.0, 3.1415, 1.626]);

// Create the mandelbrot set:
let mandelbrot = Matrix::<u8>::from_fn(600, 800, |i|{
    let x = ((i % 800) as f32 - 400.0) / 200.0; // Go from -2 to 2.
    let y = ((i / 800) as f32 - 400.0) / 200.0; // Also -2 to 2.

    let c = (x, y);
    let mut z = (0f32, 0f32);

    let mut iter = 0u8;
    for t in 0..255u8 {
        if z.0*z.0 + z.1*z.1 > 4.0 {
            break;
        }
        z = ((z.0 * z.0) - (z.1 * z.1), 2*(z.0 * z.1));
        z = (z.0 + c.0, z.1 + c.1);
        iter = t;
    }

    iter
});

// Multiply a 3x3 matrix with the identity matrix.
let a = Matrix::<f32>::ones(3, 3);
let ident = Matirx::<f32>::ident(3, 3);
let res = a.matmul(ident);
```
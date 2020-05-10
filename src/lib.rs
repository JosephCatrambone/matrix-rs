
use num_traits::Num;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Index, IndexMut, Range};

#[derive(Clone, Debug, Default)]
struct Matrix<T:Num+Send+Sync+Copy> {
	rows: usize,
	columns: usize,
	data: Vec<T>
}

impl<T:Num + Send + Sync + Copy> Matrix<T> {
	fn zeros(rows:usize, columns:usize) -> Self {
		Matrix::from_fn(rows, columns, |_| { T::zero() })
	}

	fn ones(rows:usize, columns:usize) -> Self {
		Matrix::from_fn(rows, columns, |_| { T::one() })
	}

	fn from_data(rows:usize, columns:usize, data:Vec<T>) -> Self {
		Matrix { rows, columns, data }
	}

	fn from_fn<F>(rows:usize, columns:usize, func:F) -> Self where F: Fn(usize)->T {
		Matrix {
			rows,
			columns,
			data: (0..rows*columns).map(|i|{ func(i) }).collect()
		}
	}

	//fn hstack(matrices: impl Iterator<Item=Matrix<T>> + Index<usize, Output=Matrix<T>> + Copy) -> Self {
	fn hstack(matrices: Vec<&Matrix<T>>) -> Self {
		let mut total_columns = 0;
		let rows = matrices[0].rows;
		let mut column_to_matrix_index = vec![];
		let mut matrix_start_column = vec![];
		for (mat_idx, m) in matrices.iter().enumerate() {
			for c in total_columns..(total_columns + m.columns) {
				column_to_matrix_index.push(mat_idx);
			}
			matrix_start_column.push(total_columns);
			total_columns += m.columns;
			assert_eq!(rows, m.rows);
		}
		// TODO: The lookup to find the source matrix might take longer than allocating everything and then going back and setting the values.
		Matrix::from_fn(rows, total_columns, |i|{
			let this_r = i/total_columns;
			let this_c = i%total_columns;
			let source_matrix = column_to_matrix_index[this_c];
			matrices[source_matrix].get(this_r, this_c - matrix_start_column[source_matrix])
		})
	}

	fn get_width(&self) -> usize {
		return self.columns
	}

	fn get_height(&self) -> usize {
		return self.rows
	}

	fn get(&self, iry:usize, jcx:usize) -> T {
		self.data[jcx + iry*self.columns]
	}

	fn get_row(&self, iry:usize) -> Matrix<T> {
		Matrix::from_fn(1, self.columns, |i|{
			self.get(iry, i)
		})
	}

	fn get_column(&self, jcx:usize) -> Matrix<T> {
		Matrix::from_fn(self.rows, 1, |i| {
			self.get(i, jcx)
		})
	}

	fn set(&mut self, iry:usize, jcx:usize, value:T) {
		self.data[jcx + iry*self.columns] = value;
	}

	fn set_row(&mut self, iry:usize, data: impl Iterator<Item = T>) {
		for (idx, v) in data.enumerate() {
			self.set(iry, idx, v);
		}
	}

	fn set_column(&mut self, jcx:usize, data: impl Iterator<Item = T>) {
		for (idx, v) in data.enumerate() {
			self.set(idx, jcx, v);
		}
	}

	fn swap_rows(&mut self, a:usize, b:usize) {
		for column in 0..self.columns {
			let tmp = self.get(a, column);
			self.set(a, column, self.get(b, column));
			self.set(b, column, tmp);
		}
	}

	fn swap_columns(&mut self, a:usize, b:usize) {
		for row in 0..self.rows {
			let tmp = self.get(row, a);
			self.set(row, a, self.get(row, b));
			self.set(row, b, tmp);
		}
	}

	// TODO: This may be unnecessary with new_from_fn().
	/*
	fn op<F>(&self, f:F) -> Matrix<T> where F: Fn(usize, &T) -> T + Sync {
		Matrix {
			rows: self.rows,
			columns: self.columns,
			data: self.data.par_iter().enumerate().for_each(|(idx, v)| { f(idx, &v) })
		}
	}
	*/

	fn assign_op<F>(&mut self, f:F) -> () where F: Fn(usize, &T) -> T + Sync {
		self.data.par_iter_mut().enumerate().for_each(|(idx, v)|{ *v = f(idx, &v)});
	}

	fn op<F>(&mut self, f:F) -> Matrix<T> where F: Fn(usize, &T) -> T + Sync {
		Matrix::from_data(self.rows, self.columns, self.data.par_iter_mut().enumerate().map(|(idx, v)|{ f(idx, &v) }).collect())
	}

	fn copy_slice(&self, row_start:i64, row_end:i64, column_start:i64, column_end:i64) -> Matrix<T> {
		let x_start = if column_start < 0 {
			self.columns as i64 + column_start
		} else {
			column_start
		} as usize;
		let x_end = if column_end < 0 {
			self.columns as i64 + column_end
		} else {
			column_end
		} as usize;

		let y_start = if row_start < 0 {
			self.rows as i64 + row_start
		} else {
			row_start
		} as usize;
		let y_end = if row_end < 0 {
			self.rows as i64 + row_end
		} else {
			row_end
		} as usize;

		Matrix::from_fn(y_end - y_start, x_end-x_start, |i|{
			let width = x_end - x_start;
			let y = i/width;
			let x = i%width;
			self.data[(x+x_start) + (y+y_start)*width].clone()
		})
	}

	fn matmul(&self, other:&Self) -> Self {
		assert_eq!(self.columns, other.rows);
		Matrix::<T>::from_fn(self.rows, other.columns, |i|{
			let c = i%other.columns;
			let r = i/other.columns;
			let mut accumulator = T::zero();
			for k in 0..self.columns {
				// The element located at (r, c) or (y, x) is equal to the sum of every
				accumulator = accumulator + (self.get(r, k) * other.get(k, c));
			}
			accumulator
		})
	}

	fn copy_transpose(&self) -> Matrix<T> {
		Matrix::<T>::from_fn(self.columns, self.rows, |i|{
			let old_r = i/self.columns;
			let old_c = i%self.columns;
			self.data[old_r + old_c*self.columns]
		})
	}

	fn lu_decompose(&mut self) {
		// Perform Gauss-Jordan elimination for each row from 0 to min(rows, columns).
		// If there are more columns than rows, the method will still complete.
		// This is an intentional choice, as it allows one to append an identity matrix to yield an inverse.
		for i in 0..self.rows.min(self.columns) {
			// Select the maximum value in column i.
			// Minimum would probably be numerically more stable, but has the issue of picking zero after our elimination.
			let mut max_row_idx = 0;
			let mut max_row_value = T::zero();
			let mut positive_max_row_value = T::zero();
			for j in i..self.rows {
				// Type T doesn't necessarily have .abs().
				let row_value = self.get(i, j);
				let positive_row_value = if row_value > 0 {
					row_value
				} else {
					-row_value
				};
				// If this value is biggest.
				if positive_row_value > positive_max_row_value {
					max_row_idx = j;
					max_row_value = row_value;
					positive_max_row_value = positive_row_value;
				}
			}
			// Now that we have the biggest row, normalize it so that the leading digit is 'one'.
			let inverse = 1.0/max_row_value;
			for k in i..self.columns {
				self.set(max_row_idx, k, self.get(max_row_idx, k)*inverse);
			}
			// Now that our lead is one, for all rows above and below, subtract this row.
		}
	}
}

// Accessors.
impl<T:Num + Send + Sync + Copy> Index<(usize, usize)> for Matrix<T> {
	type Output = T;

	fn index(&self, index: (usize, usize)) -> &Self::Output {
		&self.data[index.1 + index.0*self.columns]
	}
}

// Define addition, subtraction, multiplication, and division for all the things.
macro_rules! def_operator {
    ( $opname:ident, $inplaceopname:ident, $fnname:ident, $fnnameinplace:ident, $op:tt, $inplaceop:tt ) => {
    	// Define _= for Matrix<T>
		impl<T:Num+Send+Sync+Copy> $inplaceopname<Self> for Matrix<T> {
			fn $fnnameinplace(&mut self, rhs: Self) {
				self.assign_op(|i, v|{ *v $op rhs.data[i] });
			}
		}

		// Define _= for &Matrix<T>
		impl<T:Num+Send+Sync+Copy> $inplaceopname<&Self> for Matrix<T> {
			fn $fnnameinplace(&mut self, rhs: &Self) {
				self.assign_op(|i, v|{ *v $op rhs.data[i] });
			}
		}

		// Define _= for a generic number.
		impl<T:Num+Send+Sync+Copy> $inplaceopname<T> for Matrix<T> {
			fn $fnnameinplace(&mut self, rhs: T) {
				self.assign_op(|i, val|{ *val $op rhs });
			}
		}

		// Define _ for a matrix.
		impl<T:Num+Send+Sync+Copy> $opname for Matrix<T> {
			type Output = Matrix<T>;

			fn $fnname(self, rhs: Self) -> Self::Output {
				let mut new = Matrix {
					rows: self.rows,
					columns: self.columns,
					data: Vec::from(self.data)
				};
				new $inplaceop rhs;
				new
			}
		}

		// Define _ for a generic number.
		impl<T:Num+Send+Sync+Copy> $opname<T> for Matrix<T> {
			type Output = Matrix<T>;

			fn $fnname(self, rhs: T) -> Self::Output {
				self $op rhs
			}
		}
	};
}

def_operator!(Add, AddAssign, add, add_assign, +, +=);
def_operator!(Sub, SubAssign, sub, sub_assign, -, -=);
def_operator!(Mul, MulAssign, mul, mul_assign, *, *=);
def_operator!(Div, DivAssign, div, div_assign, /, /=);


#[cfg(test)]
mod tests {
	use crate::Matrix;

	#[test]
	fn it_works() {
		assert_eq!(2 + 2, 4);
	}

	#[test]
	fn row_major() {
		let m = Matrix::<u32>::from_fn(2, 3, |i|{ i as u32 });
		assert_eq!(m.data[0], 0);
		assert_eq!(m.data[1], 1);
		assert_eq!(m.data[2], 2);
		assert_eq!(m.data[3], 3);
		assert_eq!(m[(0, 0)], 0);

		let n = m.copy_slice(0, 1, 0, -1);
		assert_eq!(n.rows, 1);
		assert_eq!(n.columns, 2);

		let o = &m;
		assert_eq!(o.get(0, 0), 0);
	}

	#[test]
	fn test_in_place() {
		let mut m = Matrix::<u32>::from_fn(2, 3, |i| { i as u32 });
		m.assign_op(|_idx, v|{ (*v) * 3 });
		for i in 0..2*3 {
			assert_eq!(m.data[i], (i*3) as u32);
		}

		m += 1;
		for i in 0..2*3 {
			assert_eq!(m.data[i], 1+(i*3) as u32);
		}

		let n = m.clone();
		m += n;
		for i in 0..2*3 {
			assert_eq!(m.data[i], (2*(1+(i*3)) as u32));
		}

		let m_prev = m.clone();
		let n = Matrix::<u32>::zeros(2, 3);
		m += n;
		for i in 0..2*3 {
			assert_eq!(m.data[i], m_prev.data[i]);
		}
	}

	#[test]
	fn test_row_col_swap_and_transpose() {
		let m = Matrix::<u8>::from_fn(2, 2, |i| { i as u8 });
		let mut n = m.clone();
		n.swap_rows(0, 1);
		assert_eq!(n[(0, 0)], m[(1, 0)]);
		assert_eq!(n[(0, 1)], m[(1, 1)]);

		let t = m.copy_transpose();
		assert_eq!(t.get(0, 0), m.get(0, 0));
		assert_eq!(t.get(1, 0), m.get(0, 1));
		assert_eq!(t.get(0, 1), m.get(1, 0));
		assert_eq!(t.get(1, 1), m.get(1, 1));
	}

	#[test]
	fn test_matmul() {
		let mut m = Matrix::<u32>::ones(5, 6);
		let mut n = Matrix::<u32>::ones(6, 7);
		let res = m.matmul(&n);
		assert_eq!(res.rows, 5);
		assert_eq!(res.columns, 7);
		for i in 0..5 {
			for j in 0..7 {
				assert_eq!(res.get(i, j), 6);
			}
		}
	}

	#[test]
	fn test_hstack() {
		let a = Matrix::<f32>::from_data(3, 1, vec![1.0, 0.0, 0.0]);
		let b = Matrix::from_data(3, 1, vec![0.0, 1.0, 0.0]);
		let c = Matrix::from_data(3, 1, vec![0.0, 0.0, 1.0]);
		let ident = Matrix::hstack(vec![&a, &b, &c]);
		let foo = Matrix::<f32>::from_fn(8, 3, |i| { (i*i) as f32 });
		let res = foo.matmul(&ident);
		assert_eq!(res.rows, 8);
		assert_eq!(res.columns, 3);
		for (a,b) in foo.data.iter().zip(res.data.iter()) {
			assert_eq!(*a, *b);
		}
	}

	#[test]
	fn test_gauss_jordan_elimination() {

	}
}

module gl3n.matrix;

import gl3n.vector;
import std.traits : isFloatingPoint, isIntegral;
import std.algorithm : reduce;
import std.math : sin, cos, tan, PI;

version(NoReciprocalMul)
{
	private enum ReciprocalMul = false;
}
else
{
	private enum ReciprocalMul = true;
}

/// Base template for all matrix-types.
/// Params:
///  type = all values get stored as this type
///  rows_ = rows of the matrix
///  cols_ = columns of the matrix
/// Examples:
/// ---
/// alias Matrix!(float, 4, 4) mat4;
/// alias Matrix!(double, 3, 4) mat34d;
/// alias Matrix!(real, 2, 2) mat2r;
/// ---
struct Matrix(type, size_t rows_, size_t cols_)
	if((rows_ > 0) && (cols_ > 0))
{
	alias mt = type; /// Holds the internal type of the matrix;
	alias vt = Vector!(mt, cols_);
	enum rows = rows_; /// Holds the number of rows;
	enum cols = cols_; /// Holds the number of columns;
	
	/// Holds the matrix $(RED row-major) in memory.
	vt[rows] matrix = identityVectors;
	alias matrix this;
	
	unittest
	{
		mat2 m2 = mat2(0.0f, 1.0f, 2.0f, 3.0f);
		assert(m2[0][0] == 0.0f);
		assert(m2[0][1] == 1.0f);
		assert(m2[1][0] == 2.0f);
		assert(m2[1][1] == 3.0f);
		m2[0..1] = [2.0f, 2.0f];
		assert(m2 == [[2.0f, 2.0f], [2.0f, 3.0f]]);
		
		mat3 m3 = mat3(0.0f, 0.1f, 0.2f, 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f);
		assert(m3[0][1] == 0.1f);
		assert(m3[2][0] == 2.0f);
		assert(m3[1][2] == 1.2f);
		m3[0][0..$] = 0.0f;
		assert(m3 == [[0.0f, 0.0f, 0.0f],
			[1.0f, 1.1f, 1.2f],
			[2.0f, 2.1f, 2.2f]]);
		
		mat4 m4 = mat4(0.0f, 0.1f, 0.2f, 0.3f,
		               1.0f, 1.1f, 1.2f, 1.3f,
		               2.0f, 2.1f, 2.2f, 2.3f,
		               3.0f, 3.1f, 3.2f, 3.3f);
		assert(m4[0][3] == 0.3f);
		assert(m4[1][1] == 1.1f);
		assert(m4[2][0] == 2.0f);
		assert(m4[3][2] == 3.2f);
		m4[2][1..3] = [1.0f, 2.0f];
		assert(m4 == [[0.0f, 0.1f, 0.2f, 0.3f],
			[1.0f, 1.1f, 1.2f, 1.3f],
			[2.0f, 1.0f, 2.0f, 2.3f],
			[3.0f, 3.1f, 3.2f, 3.3f]]);
		
	}
	
	/// Returns the pointer to the stored values as OpenGL requires it.
	/// Note this will return a pointer to a $(RED row-major) matrix,
	/// $(RED this means you've to set the transpose argument to GL_TRUE when passing it to OpenGL).
	/// Examples:
	/// ---
	/// // 3rd argument = GL_TRUE
	/// glUniformMatrix4fv(programs.main.model, 1, GL_TRUE, mat4.translation(-0.5f, -0.5f, 1.0f).value_ptr);
	/// ---
	@property auto ptr()
	{
		return matrix[0].ptr;
	}
	
	/// Returns the current matrix formatted as flat string.
	@property string toString()
	{
		import std.string : format;
		return format("%s", matrix);
	}
	
	/// Returns the current matrix as pretty formatted string.
	@property string toPrettyString() {
		import std.string : format, rightJustify, join;
		string fmtr = "%s";
		
		size_t rjust = max(format(fmtr, reduce!(max)(matrix[])).length,
		                   format(fmtr, reduce!(min)(matrix[])).length) - 1;
		
		string[] outer_parts;
		foreach(mt[] row; matrix) {
			string[] inner_parts;
			foreach(mt col; row) {
				inner_parts ~= rightJustify(format(fmtr, col), rjust);
			}
			outer_parts ~= " [" ~ join(inner_parts, ", ") ~ "]";
		}
		
		return "[" ~ join(outer_parts, "\n")[1..$] ~ "]";
	}
	
	@safe pure nothrow:
	static void isCompatibleMatrixImpl(int r, int c)(Matrix!(mt, r, c) m) { }
	enum isCompatibleMatrix(T) = is(typeof(isCompatibleMatrixImpl(T.init)));
	
	static void isCompatibleVectorImpl(int d)(Vector!(mt, d) vec) { }
	enum isCompatibleVector(T) = is(typeof(isCompatibleVectorImpl(T.init)));
	
	private void construct(int i, T, Tail...)(T head, Tail tail)
	{
		static if(i >= rows*cols)
		{
			static assert(false, "Too many arguments passed to constructor");
		}
		else static if(is(T : mt))
		{
			matrix[i / cols][i % cols] = head;
			construct!(i + 1)(tail);
		}
		else static if(is(T == Vector!(mt, cols)))
		{
			static if(i % cols == 0)
			{
				matrix[i / cols] = head;
				construct!(i + T.dimension)(tail);
			}
			else
			{
				static assert(false, "Can't convert Vector into the matrix. Maybe it doesn't align to the columns correctly or dimension doesn't fit");
			}
		}
		else
		{
			static assert(false, "Matrix constructor argument must be of type " ~ mt.stringof ~ " or Vector, not " ~ T.stringof);
		}
	}
	
	private void construct(int i)()
	{ // terminate
		static assert(i == rows*cols, "Not enough arguments passed to constructor");
	}
	
	/// Constructs the matrix:
	/// If a single value is passed, the matrix will be cleared with this value (each column in each row will contain this value).
	/// If a matrix with more rows and columns is passed, the matrix will be the upper left nxm matrix.
	/// If a matrix with less rows and columns is passed, the passed matrix will be stored in the upper left of an identity matrix.
	/// It's also allowed to pass vectors and scalars at a time, but the vectors dimension must match the number of columns and align correctly.
	/// Examples:
	/// ---
	/// mat2 m2 = mat2(0.0f); // mat2 m2 = mat2(0.0f, 0.0f, 0.0f, 0.0f);
	/// mat3 m3 = mat3(m2); // mat3 m3 = mat3(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
	/// mat3 m3_2 = mat3(vec3(1.0f, 2.0f, 3.0f), 4.0f, 5.0f, 6.0f, vec3(7.0f, 8.0f, 9.0f));
	/// mat4 m4 = mat4.identity; // just an identity matrix
	/// mat3 m3_3 = mat3(m4); // mat3 m3_3 = mat3.identity
	/// ---
	this(Args...)(Args args)
	{
		construct!(0)(args);
	}
	
	/// ditto
	this(T)(T mat)
		if(isMatrix!T && (T.cols >= cols) && (T.rows >= rows))
	{
		foreach(r; TupleRange!(0, rows))
		{
			foreach(c; TupleRange!(0, cols))
			{
				matrix[r][c] = mat.matrix[r][c];
			}
		}
	}
	
	/// ditto
	this(T)(T mat)
		if(isMatrix!T && (T.cols < cols) && (T.rows < rows))
	{
		makeIdentity();
		
		foreach(r; TupleRange!(0, T.rows))
		{
			foreach(c; TupleRange!(0, T.cols))
			{
				matrix[r][c] = mat.matrix[r][c];
			}
		}
	}
	
	/// ditto
	this()(mt value)
	{
		clear(value);
	}
	
	/// Returns true if all values are not nan and finite, otherwise false.
	@property bool isFinite() const
	{
		static if(isIntegral!type)
		{
			return true;
		}
		else
		{
			import std.algorithm : all;
			return matrix[].all!(a => a.isFinite);
		}
	}
	
	/// Sets all values of the matrix to value (each column in each row will contain this value).
	void clear(mt value)
	{
		foreach(r; TupleRange!(0, rows))
		{
			matrix[r] = vt(value);
		}
	}
	
	unittest
	{
		mat2 m2 = mat2(1.0f, 1.0f, vec2(2.0f, 2.0f));
		assert(m2.matrix == [[1.0f, 1.0f], [2.0f, 2.0f]]);
		m2.clear(3.0f);
		assert(m2.matrix == [[3.0f, 3.0f], [3.0f, 3.0f]]);
		assert(m2.isFinite);
		m2.clear(float.nan);
		assert(!m2.isFinite);
		m2.clear(float.infinity);
		assert(!m2.isFinite);
		m2.clear(0.0f);
		assert(m2.isFinite);
		
		mat3 m3 = mat3(1.0f);
		assert(m3.matrix == [[1.0f, 1.0f, 1.0f],
			[1.0f, 1.0f, 1.0f],
			[1.0f, 1.0f, 1.0f]]);
		
		mat4 m4 = mat4(vec4(1.0f, 1.0f, 1.0f, 1.0f),
		               2.0f, 2.0f, 2.0f, 2.0f,
		               3.0f, 3.0f, 3.0f, 3.0f,
		               vec4(4.0f, 4.0f, 4.0f, 4.0f));
		assert(m4.matrix == [[1.0f, 1.0f, 1.0f, 1.0f],
			[2.0f, 2.0f, 2.0f, 2.0f],
			[3.0f, 3.0f, 3.0f, 3.0f],
			[4.0f, 4.0f, 4.0f, 4.0f]]);
		assert(mat3(m4).matrix == [[1.0f, 1.0f, 1.0f],
			[2.0f, 2.0f, 2.0f],
			[3.0f, 3.0f, 3.0f]]);
		assert(mat2(mat3(m4)).matrix == [[1.0f, 1.0f], [2.0f, 2.0f]]);
		assert(mat2(m4).matrix == mat2(mat3(m4)).matrix);
		assert(mat4(mat3(m4)).matrix == [[1.0f, 1.0f, 1.0f, 0.0f],
			[2.0f, 2.0f, 2.0f, 0.0f],
			[3.0f, 3.0f, 3.0f, 0.0f],
			[0.0f, 0.0f, 0.0f, 1.0f]]);
		
		Matrix!(float, 2, 3) mt1 = Matrix!(float, 2, 3)(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
		Matrix!(float, 3, 2) mt2 = Matrix!(float, 3, 2)(6.0f, -1.0f, 3.0f, 2.0f, 0.0f, -3.0f);
		
		assert(mt1.matrix == [[1.0f, 2.0f, 3.0f], [4.0f, 5.0f, 6.0f]]);
		assert(mt2.matrix == [[6.0f, -1.0f], [3.0f, 2.0f], [0.0f, -3.0f]]);
		
		static assert(!__traits(compiles, mat2(1, 2, 1)));
		static assert(!__traits(compiles, mat3(1, 2, 3, 1, 2, 3, 1, 2)));
		static assert(!__traits(compiles, mat4(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3)));
	}
	
	const(Vector!(mt, rows)) col(size_t index) const
		in
	{
		assert(index < cols);
	}
	body
	{
		Vector!(mt, rows) vec;
		
		foreach(i; TupleRange!(0, rows))
		{
			vec[i] = matrix[i][index];
		}
		
		return vec;
	}
	
	static if(rows == cols)
	{
		/// Transposes the current matrix;
		void transpose()
		{
			this = transposed;
		}
		
		unittest
		{
			mat2 m2 = mat2(1.0f);
			m2.transpose();
			assert(m2.matrix == mat2(1.0f).matrix);
			m2.makeIdentity();
			assert(m2.matrix == [[1.0f, 0.0f],
				[0.0f, 1.0f]]);
			m2.transpose();
			assert(m2.matrix == [[1.0f, 0.0f],
				[0.0f, 1.0f]]);
			assert(m2.matrix == m2.identity.matrix);
			
			mat3 m3 = mat3(1.1f, 1.2f, 1.3f,
			               2.1f, 2.2f, 2.3f,
			               3.1f, 3.2f, 3.3f);
			m3.transpose();
			assert(m3.matrix == [[1.1f, 2.1f, 3.1f],
				[1.2f, 2.2f, 3.2f],
				[1.3f, 2.3f, 3.3f]]);
			
			mat4 m4 = mat4(2.0f);
			m4.transpose();
			assert(m4.matrix == mat4(2.0f).matrix);
			m4.makeIdentity();
			assert(m4.matrix == [[1.0f, 0.0f, 0.0f, 0.0f],
				[0.0f, 1.0f, 0.0f, 0.0f],
				[0.0f, 0.0f, 1.0f, 0.0f],
				[0.0f, 0.0f, 0.0f, 1.0f]]);
			assert(m4.matrix == m4.identity.matrix);
		}
		
	}
	
	/// Returns a transposed copy of the matrix.
	@property Matrix!(mt, cols, rows) transposed() const
	{
		typeof(return) ret;
		
		foreach(c; TupleRange!(0, cols))
		{
			ret[c] = col(c);
		}
		
		return ret;
	}
	
	// transposed already tested in last unittest
	
	
	static if((rows == 2) && (cols == 2))
	{
		@property mt det() const
		{
			return (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]);
		}
		
		private Matrix invert(ref Matrix mat) const
		{
			static if(isFloatingPoint!mt && ReciprocalMul)
			{
				mt d = 1 / det;
				
				mat = Matrix(
					col(1) * d,
					-col(0) * d);
			}
			else
			{
				mt d = det;
				
				mat = Matrix(
					col(1) / d,
					-col(0) / d);
			}
			
			return mat;
		}
		
		static Matrix scaling(mt x, mt y)
		{
			Matrix ret;
			
			ret.matrix[0][0] = x;
			ret.matrix[1][1] = y;
			
			return ret;
		}
		
		Matrix scale(mt x, mt y)
		{
			this = Matrix.scaling(x, y) * this;
			return this;
		}
		
		unittest
		{
			assert(mat2.scaling(3, 3).matrix == mat2.identity.scale(3, 3).matrix);
			assert(mat2.scaling(3, 3).matrix == [[3.0f, 0.0f], [0.0f, 3.0f]]);
		}
		
	}
	else static if((rows == 3) && (cols == 3))
	{
		@property mt det() const
		{
			return (matrix[0][0] * matrix[1][1] * matrix[2][2]
			+ matrix[0][1] * matrix[1][2] * matrix[2][0]
			+ matrix[0][2] * matrix[1][0] * matrix[2][1]
			- matrix[0][2] * matrix[1][1] * matrix[2][0]
			- matrix[0][1] * matrix[1][0] * matrix[2][2]
			- matrix[0][0] * matrix[1][2] * matrix[2][1]);
		}
		
		private Matrix invert(ref Matrix mat) const
		{
			static if(isFloatingPoint!mt && ReciprocalMul)
			{
				mt d = 1 / det;
				enum op = "*";
			}
			else
			{
				mt d = det;
				enum op = "/";
			}
			
			mixin(`
            mat = Matrix((matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])`~op~`d,
                         (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2])`~op~`d,
                         (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1])`~op~`d,
                         (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2])`~op~`d,
                         (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0])`~op~`d,
                         (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2])`~op~`d,
                         (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])`~op~`d,
                         (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1])`~op~`d,
                         (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])`~op~`d);
            `);
			
			return mat;
		}
	}
	else static if((rows == 4) && (cols == 4))
	{
		/// Returns the determinant of the current matrix (2x2, 3x3 and 4x4 matrices).
		@property mt det() const
		{
			return (matrix[0][3] * matrix[1][2] * matrix[2][1] * matrix[3][0] - matrix[0][2] * matrix[1][3] * matrix[2][1] * matrix[3][0]
			- matrix[0][3] * matrix[1][1] * matrix[2][2] * matrix[3][0] + matrix[0][1] * matrix[1][3] * matrix[2][2] * matrix[3][0]
			+ matrix[0][2] * matrix[1][1] * matrix[2][3] * matrix[3][0] - matrix[0][1] * matrix[1][2] * matrix[2][3] * matrix[3][0]
			- matrix[0][3] * matrix[1][2] * matrix[2][0] * matrix[3][1] + matrix[0][2] * matrix[1][3] * matrix[2][0] * matrix[3][1]
			+ matrix[0][3] * matrix[1][0] * matrix[2][2] * matrix[3][1] - matrix[0][0] * matrix[1][3] * matrix[2][2] * matrix[3][1]
			- matrix[0][2] * matrix[1][0] * matrix[2][3] * matrix[3][1] + matrix[0][0] * matrix[1][2] * matrix[2][3] * matrix[3][1]
			+ matrix[0][3] * matrix[1][1] * matrix[2][0] * matrix[3][2] - matrix[0][1] * matrix[1][3] * matrix[2][0] * matrix[3][2]
			- matrix[0][3] * matrix[1][0] * matrix[2][1] * matrix[3][2] + matrix[0][0] * matrix[1][3] * matrix[2][1] * matrix[3][2]
			+ matrix[0][1] * matrix[1][0] * matrix[2][3] * matrix[3][2] - matrix[0][0] * matrix[1][1] * matrix[2][3] * matrix[3][2]
			- matrix[0][2] * matrix[1][1] * matrix[2][0] * matrix[3][3] + matrix[0][1] * matrix[1][2] * matrix[2][0] * matrix[3][3]
			+ matrix[0][2] * matrix[1][0] * matrix[2][1] * matrix[3][3] - matrix[0][0] * matrix[1][2] * matrix[2][1] * matrix[3][3]
			- matrix[0][1] * matrix[1][0] * matrix[2][2] * matrix[3][3] + matrix[0][0] * matrix[1][1] * matrix[2][2] * matrix[3][3]);
		}
		
		private Matrix invert(ref Matrix mat) const
		{
			static if(isFloatingPoint!mt && ReciprocalMul)
			{
				mt d = 1 / det;
				enum op = "*";
			}
			else
			{
				mt d = det;
				enum op = "/";
			}
			
			mixin(`
            mat.matrix = Matrix(
                          (matrix[1][1] * matrix[2][2] * matrix[3][3] + matrix[1][2] * matrix[2][3] * matrix[3][1] + matrix[1][3] * matrix[2][1] * matrix[3][2]
                         -matrix[1][1] * matrix[2][3] * matrix[3][2] - matrix[1][2] * matrix[2][1] * matrix[3][3] - matrix[1][3] * matrix[2][2] * matrix[3][1])`~op~`d,
                          (matrix[0][1] * matrix[2][3] * matrix[3][2] + matrix[0][2] * matrix[2][1] * matrix[3][3] + matrix[0][3] * matrix[2][2] * matrix[3][1]
                         -matrix[0][1] * matrix[2][2] * matrix[3][3] - matrix[0][2] * matrix[2][3] * matrix[3][1] - matrix[0][3] * matrix[2][1] * matrix[3][2])`~op~`d,
                          (matrix[0][1] * matrix[1][2] * matrix[3][3] + matrix[0][2] * matrix[1][3] * matrix[3][1] + matrix[0][3] * matrix[1][1] * matrix[3][2]
                         -matrix[0][1] * matrix[1][3] * matrix[3][2] - matrix[0][2] * matrix[1][1] * matrix[3][3] - matrix[0][3] * matrix[1][2] * matrix[3][1])`~op~`d,
                          (matrix[0][1] * matrix[1][3] * matrix[2][2] + matrix[0][2] * matrix[1][1] * matrix[2][3] + matrix[0][3] * matrix[1][2] * matrix[2][1]
                         -matrix[0][1] * matrix[1][2] * matrix[2][3] - matrix[0][2] * matrix[1][3] * matrix[2][1] - matrix[0][3] * matrix[1][1] * matrix[2][2])`~op~`d,
                          (matrix[1][0] * matrix[2][3] * matrix[3][2] + matrix[1][2] * matrix[2][0] * matrix[3][3] + matrix[1][3] * matrix[2][2] * matrix[3][0]
                         -matrix[1][0] * matrix[2][2] * matrix[3][3] - matrix[1][2] * matrix[2][3] * matrix[3][0] - matrix[1][3] * matrix[2][0] * matrix[3][2])`~op~`d,
                          (matrix[0][0] * matrix[2][2] * matrix[3][3] + matrix[0][2] * matrix[2][3] * matrix[3][0] + matrix[0][3] * matrix[2][0] * matrix[3][2]
                         -matrix[0][0] * matrix[2][3] * matrix[3][2] - matrix[0][2] * matrix[2][0] * matrix[3][3] - matrix[0][3] * matrix[2][2] * matrix[3][0])`~op~`d,
                          (matrix[0][0] * matrix[1][3] * matrix[3][2] + matrix[0][2] * matrix[1][0] * matrix[3][3] + matrix[0][3] * matrix[1][2] * matrix[3][0]
                         -matrix[0][0] * matrix[1][2] * matrix[3][3] - matrix[0][2] * matrix[1][3] * matrix[3][0] - matrix[0][3] * matrix[1][0] * matrix[3][2])`~op~`d,
                          (matrix[0][0] * matrix[1][2] * matrix[2][3] + matrix[0][2] * matrix[1][3] * matrix[2][0] + matrix[0][3] * matrix[1][0] * matrix[2][2]
                         -matrix[0][0] * matrix[1][3] * matrix[2][2] - matrix[0][2] * matrix[1][0] * matrix[2][3] - matrix[0][3] * matrix[1][2] * matrix[2][0])`~op~`d,
                          (matrix[1][0] * matrix[2][1] * matrix[3][3] + matrix[1][1] * matrix[2][3] * matrix[3][0] + matrix[1][3] * matrix[2][0] * matrix[3][1]
                         -matrix[1][0] * matrix[2][3] * matrix[3][1] - matrix[1][1] * matrix[2][0] * matrix[3][3] - matrix[1][3] * matrix[2][1] * matrix[3][0])`~op~`d,
                          (matrix[0][0] * matrix[2][3] * matrix[3][1] + matrix[0][1] * matrix[2][0] * matrix[3][3] + matrix[0][3] * matrix[2][1] * matrix[3][0]
                         -matrix[0][0] * matrix[2][1] * matrix[3][3] - matrix[0][1] * matrix[2][3] * matrix[3][0] - matrix[0][3] * matrix[2][0] * matrix[3][1])`~op~`d,
                          (matrix[0][0] * matrix[1][1] * matrix[3][3] + matrix[0][1] * matrix[1][3] * matrix[3][0] + matrix[0][3] * matrix[1][0] * matrix[3][1]
                         -matrix[0][0] * matrix[1][3] * matrix[3][1] - matrix[0][1] * matrix[1][0] * matrix[3][3] - matrix[0][3] * matrix[1][1] * matrix[3][0])`~op~`d,
                          (matrix[0][0] * matrix[1][3] * matrix[2][1] + matrix[0][1] * matrix[1][0] * matrix[2][3] + matrix[0][3] * matrix[1][1] * matrix[2][0]
                         -matrix[0][0] * matrix[1][1] * matrix[2][3] - matrix[0][1] * matrix[1][3] * matrix[2][0] - matrix[0][3] * matrix[1][0] * matrix[2][1])`~op~`d,
                          (matrix[1][0] * matrix[2][2] * matrix[3][1] + matrix[1][1] * matrix[2][0] * matrix[3][2] + matrix[1][2] * matrix[2][1] * matrix[3][0]
                         -matrix[1][0] * matrix[2][1] * matrix[3][2] - matrix[1][1] * matrix[2][2] * matrix[3][0] - matrix[1][2] * matrix[2][0] * matrix[3][1])`~op~`d,
                          (matrix[0][0] * matrix[2][1] * matrix[3][2] + matrix[0][1] * matrix[2][2] * matrix[3][0] + matrix[0][2] * matrix[2][0] * matrix[3][1]
                         -matrix[0][0] * matrix[2][2] * matrix[3][1] - matrix[0][1] * matrix[2][0] * matrix[3][2] - matrix[0][2] * matrix[2][1] * matrix[3][0])`~op~`d,
                          (matrix[0][0] * matrix[1][2] * matrix[3][1] + matrix[0][1] * matrix[1][0] * matrix[3][2] + matrix[0][2] * matrix[1][1] * matrix[3][0]
                         -matrix[0][0] * matrix[1][1] * matrix[3][2] - matrix[0][1] * matrix[1][2] * matrix[3][0] - matrix[0][2] * matrix[1][0] * matrix[3][1])`~op~`d,
                          (matrix[0][0] * matrix[1][1] * matrix[2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] + matrix[0][2] * matrix[1][0] * matrix[2][1]
                         -matrix[0][0] * matrix[1][2] * matrix[2][1] - matrix[0][1] * matrix[1][0] * matrix[2][2] - matrix[0][2] * matrix[1][1] * matrix[2][0])`~op~`d);
            `);
			
			return mat;
		}
		
		// some static fun ...
		// (1) glprogramming.com/red/appendixf.html - ortographic is broken!
		// (2) http://fly.cc.fer.hr/~unreal/theredbook/appendixg.html
		// (3) http://en.wikipedia.org/wiki/Orthographic_projection_(geometry)
		
		static if(isFloatingPoint!mt)
		{
			static private mt[6] cperspective(mt width, mt height, mt fov, mt near, mt far)
				in
			{
				assert(height != 0);
			}
			body
			{
				mt aspect = width/height;
				mt top = near * tan(fov*(PI/360.0));
				mt bottom = -top;
				mt right = top * aspect;
				mt left = -right;
				
				return [left, right, bottom, top, near, far];
			}
			
			/// Returns a perspective matrix (4x4 and floating-point matrices only).
			static Matrix perspective(mt width, mt height, mt fov, mt near, mt far)
			{
				mt[6] cdata = cperspective(width, height, fov, near, far);
				return perspective(cdata[0], cdata[1], cdata[2], cdata[3], cdata[4], cdata[5]);
			}
			
			/// ditto
			static Matrix perspective(mt left, mt right, mt bottom, mt top, mt near, mt far)
				in
			{
				assert(right-left != 0);
				assert(top-bottom != 0);
				assert(far-near != 0);
			}
			body
			{
				Matrix ret;
				ret.clear(0);
				
				ret.matrix[0][0] = (2*near)/(right-left);
				ret.matrix[0][2] = (right+left)/(right-left);
				ret.matrix[1][1] = (2*near)/(top-bottom);
				ret.matrix[1][2] = (top+bottom)/(top-bottom);
				ret.matrix[2][2] = -(far+near)/(far-near);
				ret.matrix[2][3] = -(2*far*near)/(far-near);
				ret.matrix[3][2] = -1;
				
				return ret;
			}
			
			/// Returns an inverse perspective matrix (4x4 and floating-point matrices only).
			static Matrix perspectiveInverse(mt width, mt height, mt fov, mt near, mt far) {
				mt[6] cdata = cperspective(width, height, fov, near, far);
				return perspectiveInverse(cdata[0], cdata[1], cdata[2], cdata[3], cdata[4], cdata[5]);
			}
			
			/// ditto
			static Matrix perspectiveInverse(mt left, mt right, mt bottom, mt top, mt near, mt far)
				in
			{
				assert(near != 0);
				assert(far != 0);
			}
			body
			{
				Matrix ret;
				ret.clear(0);
				
				ret.matrix[0][0] = (right-left)/(2*near);
				ret.matrix[0][3] = (right+left)/(2*near);
				ret.matrix[1][1] = (top-bottom)/(2*near);
				ret.matrix[1][3] = (top+bottom)/(2*near);
				ret.matrix[2][3] = -1;
				ret.matrix[3][2] = -(far-near)/(2*far*near);
				ret.matrix[3][3] = (far+near)/(2*far*near);
				
				return ret;
			}
			
			// (2) and (3) say this one is correct
			/// Returns an orthographic matrix (4x4 and floating-point matrices only).
			static Matrix orthographic(mt left, mt right, mt bottom, mt top, mt near, mt far)
				in
			{
				assert(right-left != 0);
				assert(top-bottom != 0);
				assert(far-near != 0);
			}
			body
			{
				Matrix ret;
				ret.clear(0);
				
				ret.matrix[0][0] = 2/(right-left);
				ret.matrix[0][3] = -(right+left)/(right-left);
				ret.matrix[1][1] = 2/(top-bottom);
				ret.matrix[1][3] = -(top+bottom)/(top-bottom);
				ret.matrix[2][2] = -2/(far-near);
				ret.matrix[2][3] = -(far+near)/(far-near);
				ret.matrix[3][3] = 1;
				
				return ret;
			}
			
			// (1) and (2) say this one is correct
			/// Returns an inverse ortographic matrix (4x4 and floating-point matrices only).
			static Matrix orthographicInverse(mt left, mt right, mt bottom, mt top, mt near, mt far)
			{
				Matrix ret;
				ret.clear(0);
				
				ret.matrix[0][0] = (right-left)/2;
				ret.matrix[0][3] = (right+left)/2;
				ret.matrix[1][1] = (top-bottom)/2;
				ret.matrix[1][3] = (top+bottom)/2;
				ret.matrix[2][2] = (far-near)/-2;
				ret.matrix[2][3] = (far+near)/2;
				ret.matrix[3][3] = 1;
				
				return ret;
			}
			
			/// Returns a look at matrix (4x4 and floating-point matrices only).
			static Matrix lookAt(Vector!(mt, 3) eye, Vector!(mt, 3) target, Vector!(mt, 3) up)
			{
				alias Vector!(mt, 3) vec3mt;
				vec3mt look_dir = (target - eye).normalized;
				vec3mt up_dir = up.normalized;
				
				vec3mt right_dir = cross(look_dir, up_dir).normalized;
				vec3mt perp_up_dir = cross(right_dir, look_dir);
				
				Matrix ret;
				ret.matrix[0].xyz = right_dir;
				ret.matrix[1].xyz = perp_up_dir;
				ret.matrix[2].xyz = -look_dir;
				
				ret.matrix[0][3] = -dot(eye, right_dir);
				ret.matrix[1][3] = -dot(eye, perp_up_dir);
				ret.matrix[2][3] = dot(eye, look_dir);
				
				return ret;
			}
			
			unittest
			{
				mt[6] cp = cperspective(600f, 900f, 60f, 1f, 100f);
				assert(cp[4] == 1.0f);
				assert(cp[5] == 100.0f);
				assert(cp[0] == -cp[1]);
				assert((cp[0] < -0.38489f) && (cp[0] > -0.38491f));
				assert(cp[2] == -cp[3]);
				assert((cp[2] < -0.577349f) && (cp[2] > -0.577351f));
				
				assert(mat4.perspective(600f, 900f, 60.0, 1.0, 100.0) == mat4.perspective(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5]));
				float[4][4] m4p = mat4.perspective(600f, 900f, 60.0, 1.0, 100.0).matrix;
				assert((m4p[0][0] < 2.598077f) && (m4p[0][0] > 2.598075f));
				assert(m4p[0][2] == 0.0f);
				assert((m4p[1][1] < 1.732052) && (m4p[1][1] > 1.732050));
				assert(m4p[1][2] == 0.0f);
				assert((m4p[2][2] < -1.020201) && (m4p[2][2] > -1.020203));
				assert((m4p[2][3] < -2.020201) && (m4p[2][3] > -2.020203));
				assert((m4p[3][2] < -0.9f) && (m4p[3][2] > -1.1f));
				
				float[4][4] m4pi = mat4.perspectiveInverse(600f, 900f, 60.0, 1.0, 100.0).matrix;
				assert((m4pi[0][0] < 0.384901) && (m4pi[0][0] > 0.384899));
				assert(m4pi[0][3] == 0.0f);
				assert((m4pi[1][1] < 0.577351) && (m4pi[1][1] > 0.577349));
				assert(m4pi[1][3] == 0.0f);
				assert(m4pi[2][3] == -1.0f);
				assert((m4pi[3][2] < -0.494999) && (m4pi[3][2] > -0.495001));
				assert((m4pi[3][3] < 0.505001) && (m4pi[3][3] > 0.504999));
				
				// maybe the next tests should be improved
				float[4][4] m4o = mat4.orthographic(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f).matrix;
				assert(m4o == [[1.0f, 0.0f, 0.0f, 0.0f],
					[0.0f, 1.0f, 0.0f, 0.0f],
					[0.0f, 0.0f, -1.0f, 0.0f],
					[0.0f, 0.0f, 0.0f, 1.0f]]);
				
				float[4][4] m4oi = mat4.orthographicInverse(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f).matrix;
				assert(m4oi == [[1.0f, 0.0f, 0.0f, 0.0f],
					[0.0f, 1.0f, 0.0f, 0.0f],
					[0.0f, 0.0f, -1.0f, 0.0f],
					[0.0f, 0.0f, 0.0f, 1.0f]]);
				
				//TODO: lookAt tests
			}
		}
	}
	
	static if((rows == cols) && (rows >= 3) && (rows <= 4))
	{
		/// Returns a translation matrix (3x3 and 4x4 matrices).
		static Matrix translation(mt x, mt y, mt z)
		{
			Matrix ret;
			
			ret.matrix[0][cols-1] = x;
			ret.matrix[1][cols-1] = y;
			ret.matrix[2][cols-1] = z;
			
			return ret;
		}
		
		/// Applys a translation on the current matrix and returns $(I this) (3x3 and 4x4 matrices).
		Matrix translate(mt x, mt y, mt z)
		{
			this = Matrix.translation(x, y, z) * this;
			return this;
		}
		
		/// Returns a scaling matrix (3x3 and 4x4 matrices);
		static Matrix scaling(mt x, mt y, mt z)
		{
			Matrix ret;
			
			ret.matrix[0][0] = x;
			ret.matrix[1][1] = y;
			ret.matrix[2][2] = z;
			
			return ret;
		}
		
		/// Applys a scale to the current matrix and returns $(I this) (3x3 and 4x4 matrices).
		Matrix scale(mt x, mt y, mt z)
		{
			this = Matrix.scaling(x, y, z) * this;
			return this;
		}
		
		unittest
		{
			mat3 m3 = mat3(1.0f);
			assert(m3.translation(1.0f, 2.0f, 3.0f).matrix == mat3.translation(1.0f, 2.0f, 3.0f).matrix);
			assert(mat3.translation(1.0f, 2.0f, 3.0f).matrix == [[1.0f, 0.0f, 1.0f],
				[0.0f, 1.0f, 2.0f],
				[0.0f, 0.0f, 3.0f]]);
			assert(mat3.identity.translate(0.0f, 1.0f, 2.0f).matrix == mat3.translation(0.0f, 1.0f, 2.0f).matrix);
			
			assert(m3.scaling(0.0f, 1.0f, 2.0f).matrix == mat3.scaling(0.0f, 1.0f, 2.0f).matrix);
			assert(mat3.scaling(0.0f, 1.0f, 2.0f).matrix == [[0.0f, 0.0f, 0.0f],
				[0.0f, 1.0f, 0.0f],
				[0.0f, 0.0f, 2.0f]]);
			assert(mat3.identity.scale(0.0f, 1.0f, 2.0f).matrix == mat3.scaling(0.0f, 1.0f, 2.0f).matrix);
			
			// same tests for 4x4
			
			mat4 m4 = mat4(1.0f);
			assert(m4.translation(1.0f, 2.0f, 3.0f).matrix == mat4.translation(1.0f, 2.0f, 3.0f).matrix);
			assert(mat4.translation(1.0f, 2.0f, 3.0f).matrix == [[1.0f, 0.0f, 0.0f, 1.0f],
				[0.0f, 1.0f, 0.0f, 2.0f],
				[0.0f, 0.0f, 1.0f, 3.0f],
				[0.0f, 0.0f, 0.0f, 1.0f]]);
			assert(mat4.identity.translate(0.0f, 1.0f, 2.0f).matrix == mat4.translation(0.0f, 1.0f, 2.0f).matrix);
			
			assert(m4.scaling(0.0f, 1.0f, 2.0f).matrix == mat4.scaling(0.0f, 1.0f, 2.0f).matrix);
			assert(mat4.scaling(0.0f, 1.0f, 2.0f).matrix == [[0.0f, 0.0f, 0.0f, 0.0f],
				[0.0f, 1.0f, 0.0f, 0.0f],
				[0.0f, 0.0f, 2.0f, 0.0f],
				[0.0f, 0.0f, 0.0f, 1.0f]]);
			assert(mat4.identity.scale(0.0f, 1.0f, 2.0f).matrix == mat4.scaling(0.0f, 1.0f, 2.0f).matrix);
		}
	}
	
	
	static if((rows == cols) && (rows >= 3) && isFloatingPoint!mt)
	{
		/// Returns an identity matrix with an applied rotateAxis around an arbitrary axis (nxn matrices, n >= 3).
		static Matrix rotation(real alpha, Vector!(mt, 3) axis)
		{
			Matrix mult;
			
			if(axis.magnitude != 1)
			{
				axis.normalize();
			}
			
			real cosa = cos(alpha);
			real sina = sin(alpha);
			
			Vector!(mt, 3) temp = (1 - cosa)*axis;
			
			mult.matrix[0][0] = cosa + temp.x * axis.x;
			mult.matrix[0][1] =        temp.x * axis.y + sina * axis.z;
			mult.matrix[0][2] =        temp.x * axis.z - sina * axis.y;
			mult.matrix[1][0] =        temp.y * axis.x - sina * axis.z;
			mult.matrix[1][1] = cosa + temp.y * axis.y;
			mult.matrix[1][2] =        temp.y * axis.z + sina * axis.x;
			mult.matrix[2][0] =        temp.z * axis.x + sina * axis.y;
			mult.matrix[2][1] =        temp.z * axis.y - sina * axis.x;
			mult.matrix[2][2] = cosa + temp.z * axis.z;
			
			return mult;
		}
		
		/// ditto
		static Matrix rotation(real alpha, mt x, mt y, mt z)
		{
			return Matrix.rotation(alpha, Vector!(mt, 3)(x, y, z));
		}
		
		/// Returns an identity matrix with an applied rotation around the x-axis (nxn matrices, n >= 3).
		static Matrix xRotation(real alpha)
		{
			Matrix mult;
			
			mt cosamt = cos(alpha);
			mt sinamt = sin(alpha);
			
			mult.matrix[1][1] = cosamt;
			mult.matrix[1][2] = -sinamt;
			mult.matrix[2][1] = sinamt;
			mult.matrix[2][2] = cosamt;
			
			return mult;
		}
		
		/// Returns an identity matrix with an applied rotation around the y-axis (nxn matrices, n >= 3).
		static Matrix yRotation(real alpha)
		{
			Matrix mult;
			
			mt cosamt = cos(alpha);
			mt sinamt = sin(alpha);
			
			mult.matrix[0][0] = cosamt;
			mult.matrix[0][2] = sinamt;
			mult.matrix[2][0] = -sinamt;
			mult.matrix[2][2] = cosamt;
			
			return mult;
		}
		
		/// Returns an identity matrix with an applied rotation around the z-axis (nxn matrices, n >= 3).
		static Matrix zRotation(real alpha)
		{
			Matrix mult;
			
			mt cosamt = cos(alpha);
			mt sinamt = sin(alpha);
			
			mult.matrix[0][0] = cosamt;
			mult.matrix[0][1] = -sinamt;
			mult.matrix[1][0] = sinamt;
			mult.matrix[1][1] = cosamt;
			
			return mult;
		}
		
		Matrix rotate(real alpha, Vector!(mt, 3) axis)
		{
			this = rotation(alpha, axis) * this;
			return this;
		}
		
		/// Rotates the current matrix around the x-axis and returns $(I this) (nxn matrices, n >= 3).
		Matrix rotateX(real alpha)
		{
			this = xRotation(alpha) * this;
			return this;
		}
		
		/// Rotates the current matrix around the y-axis and returns $(I this) (nxn matrices, n >= 3).
		Matrix rotateY(real alpha)
		{
			this = yRotation(alpha) * this;
			return this;
		}
		
		/// Rotates the current matrix around the z-axis and returns $(I this) (nxn matrices, n >= 3).
		Matrix rotateZ(real alpha)
		{
			this = zRotation(alpha) * this;
			return this;
		}
		
		unittest
		{
			assert(mat4.xRotation(0).matrix == [[1.0f, 0.0f, 0.0f, 0.0f],
				[0.0f, 1.0f, -0.0f, 0.0f],
				[0.0f, 0.0f, 1.0f, 0.0f],
				[0.0f, 0.0f, 0.0f, 1.0f]]);
			assert(mat4.yRotation(0).matrix == [[1.0f, 0.0f, 0.0f, 0.0f],
				[0.0f, 1.0f, 0.0f, 0.0f],
				[0.0f, 0.0f, 1.0f, 0.0f],
				[0.0f, 0.0f, 0.0f, 1.0f]]);
			assert(mat4.zRotation(0).matrix == [[1.0f, -0.0f, 0.0f, 0.0f],
				[0.0f, 1.0f, 0.0f, 0.0f],
				[0.0f, 0.0f, 1.0f, 0.0f],
				[0.0f, 0.0f, 0.0f, 1.0f]]);
			mat4 xro = mat4.identity;
			xro.rotateX(0);
			assert(mat4.xRotation(0).matrix == xro.matrix);
			assert(xro.matrix == mat4.identity.rotateX(0).matrix);
			assert(xro.matrix == mat4.rotation(0, vec3(1.0f, 0.0f, 0.0f)).matrix);
			mat4 yro = mat4.identity;
			yro.rotateY(0);
			assert(mat4.yRotation(0).matrix == yro.matrix);
			assert(yro.matrix == mat4.identity.rotateY(0).matrix);
			assert(yro.matrix == mat4.rotation(0, vec3(0.0f, 1.0f, 0.0f)).matrix);
			mat4 zro = mat4.identity;
			xro.rotateZ(0);
			assert(mat4.zRotation(0).matrix == zro.matrix);
			assert(zro.matrix == mat4.identity.rotateZ(0).matrix);
			assert(zro.matrix == mat4.rotation(0, vec3(0.0f, 0.0f, 1.0f)).matrix);
		}
	}
	
	static if((rows == cols) && (rows >= 2) && (rows <= 4))
	{
		/// Returns an inverted copy of the current matrix (nxn matrices, 2 >= n <= 4).
		@property Matrix inverse() const
		{
			Matrix mat;
			invert(mat);
			return mat;
		}
		
		/// Inverts the current matrix (nxn matrices, 2 >= n <= 4).
		void invert()
		{
			// workaround Issue #11238
			// uses a temporary instead of invert(this)
			Matrix temp;
			invert(temp);
			this.matrix = temp.matrix;
		}
	}
	
	unittest
	{
		mat2 m2 = mat2(1.0f, 2.0f, vec2(3.0f, 4.0f));
		assert(m2.det == -2.0f);
		assert(m2.inverse.matrix == [[-2.0f, 1.0f], [1.5f, -0.5f]]);
		
		mat3 m3 = mat3(1.0f, -2.0f, 3.0f,
		               7.0f, -1.0f, 0.0f,
		               3.0f, 2.0f, -4.0f);
		assert(m3.det == -1.0f);
		assert(m3.inverse.matrix == [[-4.0f, 2.0f, -3.0f],
			[-28.0f, 13.0f, -21.0f],
			[-17.0f, 8.0f, -13.0f]]);
		
		mat4 m4 = mat4(1.0f, 2.0f, 3.0f, 4.0f,
		               -2.0f, 1.0f, 5.0f, -2.0f,
		               2.0f, -1.0f, 7.0f, 1.0f,
		               3.0f, -3.0f, 2.0f, 0.0f);
		assert(m4.det == -8.0f);
		assert(m4.inverse.matrix == [[6.875f, 7.875f, -11.75f, 11.125f],
			[6.625f, 7.625f, -11.25f, 10.375f],
			[-0.375f, -0.375f, 0.75f, -0.625f],
			[-4.5f, -5.5f, 8.0f, -7.5f]]);
	}
	
	private void mms(mt inp, ref Matrix mat) const
	{ // mat * scalar
		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < cols; c++)
			{
				mat.matrix[r][c] = matrix[r][c] * inp;
			}
		}
	}
	
	private void masm(string op)(Matrix inp, ref Matrix mat) const
	{ // mat + or - mat
		foreach(r; TupleRange!(0, rows))
		{
			foreach(c; TupleRange!(0, cols))
			{
				mat.matrix[r][c] = mixin("inp.matrix[r][c]" ~ op ~ "matrix[r][c]");
			}
		}
	}
	
	Matrix!(mt, rows, T.cols) opBinary(string op : "*", T)(T inp) const
		if(isCompatibleMatrix!T && (T.rows == cols))
	{
		Matrix!(mt, rows, T.cols) ret;
		
		foreach(r; TupleRange!(0, rows))
		{
			foreach(c; TupleRange!(0, T.cols))
			{
				ret.matrix[r][c] = 0;
				
				foreach(c2; TupleRange!(0, cols))
				{
					ret.matrix[r][c] += matrix[r][c2] * inp.matrix[c2][c];
				}
			}
		}
		
		return ret;
	}
	
	Vector!(mt, rows) opBinary(string op : "*", T : Vector!(mt, cols))(T inp) const
	{
		Vector!(mt, rows) ret;
		ret.clear(0);
		
		foreach(c; TupleRange!(0, cols))
		{
			foreach(r; TupleRange!(0, rows))
			{
				ret.vector[r] += matrix[r][c] * inp.vector[c];
			}
		}
		
		return ret;
	}
	
	Matrix opBinary(string op : "*")(mt inp) const
	{
		Matrix ret;
		mms(inp, ret);
		return ret;
	}
	
	Matrix opBinaryRight(string op : "*")(mt inp) const
	{
		return this.opBinary!(op)(inp);
	}
	
	Matrix opBinary(string op)(Matrix inp) const if((op == "+") || (op == "-"))
	{
		Matrix ret;
		masm!(op)(inp, ret);
		return ret;
	}
	
	void opOpAssign(string op : "*")(mt inp)
	{
		mms(inp, this);
	}
	
	void opOpAssign(string op)(Matrix inp) if((op == "+") || (op == "-"))
	{
		masm!(op)(inp, this);
	}
	
	unittest
	{
		mat2 m2 = mat2(1.0f, 2.0f, 3.0f, 4.0f);
		vec2 v2 = vec2(2.0f, 2.0f);
		assert((m2*2).matrix == [[2.0f, 4.0f], [6.0f, 8.0f]]);
		assert((2*m2).matrix == (m2*2).matrix);
		m2 *= 2;
		assert(m2.matrix == [[2.0f, 4.0f], [6.0f, 8.0f]]);
		assert((m2*v2).vector == [12.0f, 28.0f]);
		assert((v2*m2).vector == [16.0f, 24.0f]);
		assert((m2*m2).matrix == [[28.0f, 40.0f], [60.0f, 88.0f]]);
		assert((m2-m2).matrix == [[0.0f, 0.0f], [0.0f, 0.0f]]);
		assert((m2+m2).matrix == [[4.0f, 8.0f], [12.0f, 16.0f]]);
		m2 += m2;
		assert(m2.matrix == [[4.0f, 8.0f], [12.0f, 16.0f]]);
		m2 -= m2;
		assert(m2.matrix == [[0.0f, 0.0f], [0.0f, 0.0f]]);
		
		mat3 m3 = mat3(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
		vec3 v3 = vec3(2.0f, 2.0f, 2.0f);
		assert((m3*2).matrix == [[2.0f, 4.0f, 6.0f], [8.0f, 10.0f, 12.0f], [14.0f, 16.0f, 18.0f]]);
		assert((2*m3).matrix == (m3*2).matrix);
		m3 *= 2;
		assert(m3.matrix == [[2.0f, 4.0f, 6.0f], [8.0f, 10.0f, 12.0f], [14.0f, 16.0f, 18.0f]]);
		assert((m3*v3).vector == [24.0f, 60.0f, 96.0f]);
		assert((v3*m3).vector == [48.0f, 60.0f, 72.0f]);
		assert((m3*m3).matrix == [[120.0f, 144.0f, 168.0f], [264.0f, 324.0f, 384.0f], [408.0f, 504.0f, 600.0f]]);
		assert((m3-m3).matrix == [[0.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]]);
		assert((m3+m3).matrix == [[4.0f, 8.0f, 12.0f], [16.0f, 20.0f, 24.0f], [28.0f, 32.0f, 36.0f]]);
		m3 += m3;
		assert(m3.matrix == [[4.0f, 8.0f, 12.0f], [16.0f, 20.0f, 24.0f], [28.0f, 32.0f, 36.0f]]);
		m3 -= m3;
		assert(m3.matrix == [[0.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]]);
		
		//TODO: tests for mat4, mat34
	}
	
	bool opCast(T : bool)() const
	{
		return isFinite;
	}
	
	unittest
	{
		assert(mat2(1.0f, 2.0f, 1.0f, 1.0f) == mat2(1.0f, 2.0f, 1.0f, 1.0f));
		assert(mat2(1.0f, 2.0f, 1.0f, 1.0f) != mat2(1.0f, 1.0f, 1.0f, 1.0f));
		
		assert(mat3(1.0f) == mat3(1.0f));
		assert(mat3(1.0f) != mat3(2.0f));
		
		assert(mat4(1.0f) == mat4(1.0f));
		assert(mat4(1.0f) != mat4(2.0f));
		
		assert(!(mat4(float.nan)));
		if(mat4(1.0f)) { }
		else { assert(false); }
	}
	
	
	private static @property vt[rows] identityVectors()
	{
		vt[rows] vectors;
		foreach(i; 0..min(cols, rows))
		{
			vectors[i][i] = 1;
		}
		return vectors;
	}
	
}

/// Pre-defined matrix types, the first number represents the number of rows
/// and the second the number of columns, if there's just one it's a nxn matrix.
/// All of these matrices are floating-point matrices.
alias Matrix!(float, 2, 2) mat2;
alias Matrix!(float, 3, 3) mat3;
alias Matrix!(float, 3, 4) mat34;
alias Matrix!(float, 4, 4) mat4;

private unittest
{
	Matrix!(float,  1, 1) A = 1;
	Matrix!(double, 1, 1) B = 1;
	Matrix!(real,   1, 1) C = 1;
	Matrix!(int,    1, 1) D = 1;
	Matrix!(float,  5, 1) E = 1;
	Matrix!(double, 5, 1) F = 1;
	Matrix!(real,   5, 1) G = 1;
	Matrix!(int,    5, 1) H = 1;
	Matrix!(float,  1, 5) I = 1;
	Matrix!(double, 1, 5) J = 1;
	Matrix!(real,   1, 5) K = 1;
	Matrix!(int,    1, 5) L = 1;
}

enum isMatrix(T)		= is(T == Matrix!Args,		Args...);
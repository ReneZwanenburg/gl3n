/**
gl3n.linalg

Special thanks to:
$(UL
  $(LI Tomasz Stachowiak (h3r3tic): allowed me to use parts of $(LINK2 https://bitbucket.org/h3r3tic/boxen/src/default/src/xf/omg, omg).)
  $(LI Jakob Øvrum (jA_cOp): improved the code a lot!)
  $(LI Florian Boesch (___doc__): helps me to understand opengl/complex maths better, see: $(LINK http://codeflow.org/).)
  $(LI #D on freenode: answered general questions about D.)
)

Authors: David Herberth
License: MIT

Note: All methods marked with pure are weakly pure since, they all access an instance member.
All static methods are strongly pure.
*/


module gl3n.linalg;

import std.math : isFinite;
import std.conv : to;
import std.traits : isIntegral, isFloatingPoint, isStaticArray, isDynamicArray, isImplicitlyConvertible, isArray;
import std.string : format, rightJustify;
import std.array : join;
import std.algorithm : max, min, reduce, all, among;
import std.functional : binaryFun;
import gl3n.math : clamp, PI, sqrt, sin, cos, acos, tan, asin, atan2, almostEqual;
import gl3n.util : isMatrix, isQuaternion, TupleRange;
import gl3n.vector;

version(NoReciprocalMul)
{
    private enum rmul = false;
}
else
{
    private enum rmul = true;
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
        return format("%s", matrix);
    }

    /// Returns the current matrix as pretty formatted string.
	@property string toPrettyString() {
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
            static if(isFloatingPoint!mt && rmul)
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
            static if(isFloatingPoint!mt && rmul)
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
            static if(isFloatingPoint!mt && rmul)
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

            mult.matrix[0][0] = to!mt(cosa + temp.x * axis.x);
            mult.matrix[0][1] = to!mt(       temp.x * axis.y + sina * axis.z);
            mult.matrix[0][2] = to!mt(       temp.x * axis.z - sina * axis.y);
            mult.matrix[1][0] = to!mt(       temp.y * axis.x - sina * axis.z);
            mult.matrix[1][1] = to!mt(cosa + temp.y * axis.y);
            mult.matrix[1][2] = to!mt(       temp.y * axis.z + sina * axis.x);
            mult.matrix[2][0] = to!mt(       temp.z * axis.x + sina * axis.y);
            mult.matrix[2][1] = to!mt(       temp.z * axis.y - sina * axis.x);
            mult.matrix[2][2] = to!mt(cosa + temp.z * axis.z);

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

            mt cosamt = to!mt(cos(alpha));
            mt sinamt = to!mt(sin(alpha));

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

            mt cosamt = to!mt(cos(alpha));
            mt sinamt = to!mt(sin(alpha));

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

            mt cosamt = to!mt(cos(alpha));
            mt sinamt = to!mt(sin(alpha));

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

/// Base template for all quaternion-types.
/// Params:
///  type = all values get stored as this type
struct Quaternion(type)
{
    alias type qt; /// Holds the internal type of the quaternion.

	union
	{
    	qt[4] quaternion; /// Holds the w, x, y and z coordinates.

		struct
		{
			qt w, x, y, z;
		}
	}

    /// Returns a pointer to the quaternion in memory, it starts with the w coordinate.
    @property auto ptr()
	{
		return quaternion.ptr;
	}

    /// Returns the current vector formatted as string, useful for printing the quaternion.
	@property string toString()
	{
        return format("%s", quaternion);
    }

    /// Constructs the quaternion.
    /// Takes a 4-dimensional vector, where vector.x = the quaternions w coordinate,
    /// or a w coordinate of type $(I qt) and a 3-dimensional vector representing the imaginary part,
    /// or 4 values of type $(I qt).
    this(qt w_, qt x_, qt y_, qt z_)
	{
        w = w_;
        x = x_;
        y = y_;
        z = z_;
    }

    /// ditto
    this(qt w_, Vector!(qt, 3) vec)
	{
        w = w_;
        quaternion[1..4] = vec.vector[];
    }

    /// ditto
    this(Vector!(qt, 4) vec)
	{
        quaternion[] = vec.vector[];
    }

    /// Returns true if all values are not nan and finite, otherwise false.
    @property bool isFinite() const
	{
		return quaternion[].all!(.isFinite);
    }

    unittest
	{
        quat q1 = quat(0.0f, 0.0f, 0.0f, 1.0f);
        assert(q1.quaternion == [0.0f, 0.0f, 0.0f, 1.0f]);
        assert(q1.quaternion == quat(0.0f, 0.0f, 0.0f, 1.0f).quaternion);
        assert(q1.quaternion == quat(0.0f, vec3(0.0f, 0.0f, 1.0f)).quaternion);
        assert(q1.quaternion == quat(vec4(0.0f, 0.0f, 0.0f, 1.0f)).quaternion);

        assert(q1.isFinite);
        q1.x = float.infinity;
        assert(!q1.isFinite);
        q1.x = float.nan;
        assert(!q1.isFinite);
        q1.x = 0.0f;
        assert(q1.isFinite);
    }

    /// Returns the squared magnitude of the quaternion.
    @property real sqrMagnitude() const
	{
        return to!real(w^^2 + x^^2 + y^^2 + z^^2);
    }

    /// Returns the magnitude of the quaternion.
    @property real magnitude() const
	{
        return sqrt(sqrMagnitude);
    }

    /// Returns an identity quaternion (w=1, x=0, y=0, z=0).
    static @property Quaternion identity()
	{
        return Quaternion(1, 0, 0, 0);
    }

    /// Makes the current quaternion an identity quaternion.
    void makeIdentity()
	{
        w = 1;
        x = 0;
        y = 0;
        z = 0;
    }

    /// Inverts the quaternion.
    void invert()
	{
        x = -x;
        y = -y;
        z = -z;
    }
    alias invert conjugate; /// ditto

    /// Returns an inverted copy of the current quaternion.
    @property Quaternion inverse() const
	{
        return Quaternion(w, -x, -y, -z);
    }
    alias inverse conjugated; /// ditto

    unittest
	{
        quat q1 = quat(1.0f, 1.0f, 1.0f, 1.0f);

        assert(q1.magnitude == 2.0f);
        assert(q1.sqrMagnitude == 4.0f);
        assert(q1.magnitude == quat(0.0f, 0.0f, 2.0f, 0.0f).magnitude);

        quat q2 = quat.identity;
        assert(q2.quaternion == [1.0f, 0.0f, 0.0f, 0.0f]);
        assert(q2.x == 0.0f);
        assert(q2.y == 0.0f);
        assert(q2.z == 0.0f);
        assert(q2.w == 1.0f);

        assert(q1.inverse.quaternion == [1.0f, -1.0f, -1.0f, -1.0f]);
        q1.invert();
        assert(q1.quaternion == [1.0f, -1.0f, -1.0f, -1.0f]);

        q1.makeIdentity();
        assert(q1.quaternion == q2.quaternion);

    }

    /// Creates a quaternion from a 3x3 matrix.
    /// Params:
    ///  matrix = 3x3 matrix (rotation)
    /// Returns: A quaternion representing the rotation (3x3 matrix)
    static Quaternion fromMatrix(Matrix!(qt, 3, 3) matrix)
	{
        Quaternion ret;

        auto mat = matrix.matrix;
        qt trace = mat[0][0] + mat[1][1] + mat[2][2];

        if(trace > 0)
		{
            real s = 0.5 / sqrt(trace + 1.0f);

            ret.w = to!qt(0.25 / s);
            ret.x = to!qt((mat[2][1] - mat[1][2]) * s);
            ret.y = to!qt((mat[0][2] - mat[2][0]) * s);
            ret.z = to!qt((mat[1][0] - mat[0][1]) * s);
        }
		else if((mat[0][0] > mat[1][1]) && (mat[0][0] > mat[2][2]))
		{
            real s = 2.0 * sqrt(1.0 + mat[0][0] - mat[1][1] - mat[2][2]);

            ret.w = to!qt((mat[2][1] - mat[1][2]) / s);
            ret.x = to!qt(0.25f * s);
            ret.y = to!qt((mat[0][1] + mat[1][0]) / s);
            ret.z = to!qt((mat[0][2] + mat[2][0]) / s);
        }
		else if(mat[1][1] > mat[2][2])
		{
            real s = 2.0 * sqrt(1 + mat[1][1] - mat[0][0] - mat[2][2]);

            ret.w = to!qt((mat[0][2] - mat[2][0]) / s);
            ret.x = to!qt((mat[0][1] + mat[1][0]) / s);
            ret.y = to!qt(0.25f * s);
            ret.z = to!qt((mat[1][2] + mat[2][1]) / s);
        }
		else
		{
            real s = 2.0 * sqrt(1 + mat[2][2] - mat[0][0] - mat[1][1]);

            ret.w = to!qt((mat[1][0] - mat[0][1]) / s);
            ret.x = to!qt((mat[0][2] + mat[2][0]) / s);
            ret.y = to!qt((mat[1][2] + mat[2][1]) / s);
            ret.z = to!qt(0.25f * s);
        }

        return ret;
    }

    /// Returns the quaternion as matrix.
    /// Params:
    ///  rows = number of rows of the resulting matrix (min 3)
    ///  cols = number of columns of the resulting matrix (min 3)
    Matrix!(qt, rows, cols) toMatrix(int rows, int cols)() const
	if((rows >= 3) && (cols >= 3))
	{
        static if((rows == 3) && (cols == 3))
		{
            Matrix!(qt, rows, cols) ret;
        }
		else
		{
            Matrix!(qt, rows, cols) ret = Matrix!(qt, rows, cols).identity;
        }

        qt xx = x^^2;
        qt xy = x * y;
        qt xz = x * z;
        qt xw = x * w;
        qt yy = y^^2;
        qt yz = y * z;
        qt yw = y * w;
        qt zz = z^^2;
        qt zw = z * w;

        ret.matrix[0][0..3] = [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)];
        ret.matrix[1][0..3] = [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)];
        ret.matrix[2][0..3] = [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)];

        return ret;
    }

    unittest
	{
        quat q1 = quat(4.0f, 1.0f, 2.0f, 3.0f);

        assert(q1.toMatrix!(3, 3).matrix == [[-25.0f, -20.0f, 22.0f], [28.0f, -19.0f, 4.0f], [-10.0f, 20.0f, -9.0f]]);
        assert(q1.toMatrix!(4, 4).matrix == [[-25.0f, -20.0f, 22.0f, 0.0f],
                                              [28.0f, -19.0f, 4.0f, 0.0f],
                                              [-10.0f, 20.0f, -9.0f, 0.0f],
                                              [0.0f, 0.0f, 0.0f, 1.0f]]);
        assert(quat.identity.toMatrix!(3, 3).matrix == Matrix!(qt, 3, 3).identity.matrix);
        assert(q1.quaternion == quat.fromMatrix(q1.toMatrix!(3, 3)).quaternion);

        assert(quat(1.0f, 0.0f, 0.0f, 0.0f).quaternion == quat.fromMatrix(mat3.identity).quaternion);

        quat q2 = quat.fromMatrix(mat3(1.0f, 3.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
        assert(q2.x == 0.0f);
        assert(almostEqual(q2.y, 0.7071067f));
        assert(almostEqual(q2.z, -1.060660));
        assert(almostEqual(q2.w, 0.7071067f));
    }

    /// Normalizes the current quaternion.
    void normalize()
	{
        qt m = to!qt(magnitude);

        if(m != 0)
		{
            w = w / m;
            x = x / m;
            y = y / m;
            z = z / m;
        }
    }

    /// Returns a normalized copy of the current quaternion.
    Quaternion normalized() const
	{
        Quaternion ret;
        qt m = to!qt(magnitude);

        if(m != 0)
		{
            ret.w = w / m;
            ret.x = x / m;
            ret.y = y / m;
            ret.z = z / m;
        }
		else
		{
            ret = Quaternion(w, x, y, z);
        }

        return ret;
    }

    unittest
	{
        quat q1 = quat(1.0f, 2.0f, 3.0f, 4.0f);
        quat q2 = quat(1.0f, 2.0f, 3.0f, 4.0f);

        q1.normalize();
        assert(q1.quaternion == q2.normalized.quaternion);
        //assert(q1.quaternion == q1.normalized.quaternion);
        assert(almostEqual(q1.magnitude, 1.0));
    }

    /// Returns the yaw.
    @property real yaw() const
	{
        return atan2(to!real(2 * (w*y + x*z)), to!real(w^^2 - x^^2 - y^^2 + z^^2));
    }

    /// Returns the pitch.
    @property real pitch() const
	{
        return asin(to!real(2 * (w*x - y*z)));
    }

    /// Returns the roll.
    @property real roll() const
	{
        return atan2(to!real(2 * (w*z + x*y)), to!real(w^^2 - x^^2 + y^^2 - z^^2));
    }

    unittest
	{
        quat q1 = quat.identity;
        assert(q1.pitch == 0.0f);
        assert(q1.yaw == 0.0f);
        assert(q1.roll == 0.0f);

        quat q2 = quat(1.0f, 1.0f, 1.0f, 1.0f);
        assert(almostEqual(q2.yaw, q2.roll));
        assert(almostEqual(q2.yaw, 1.570796f));
        assert(q2.pitch == 0.0f);

        quat q3 = quat(0.1f, 1.9f, 2.1f, 1.3f);
        assert(almostEqual(q3.yaw, 2.4382f));
        assert(isNaN(q3.pitch));
        assert(almostEqual(q3.roll, 1.67719f));
    }

    /// Returns a quaternion with applied rotation around the x-axis.
    static Quaternion xRotation(real alpha)
	{
        Quaternion ret;

        alpha /= 2;
        ret.w = to!qt(cos(alpha));
        ret.x = to!qt(sin(alpha));
        ret.y = 0;
        ret.z = 0;

        return ret;
    }

    /// Returns a quaternion with applied rotation around the y-axis.
    static Quaternion yRotation(real alpha)
	{
        Quaternion ret;

        alpha /= 2;
        ret.w = to!qt(cos(alpha));
        ret.x = 0;
        ret.y = to!qt(sin(alpha));
        ret.z = 0;

        return ret;
    }

    /// Returns a quaternion with applied rotation around the z-axis.
    static Quaternion zRotation(real alpha)
	{
        Quaternion ret;

        alpha /= 2;
        ret.w = to!qt(cos(alpha));
        ret.x = 0;
        ret.y = 0;
        ret.z = to!qt(sin(alpha));

        return ret;
    }

    /// Returns a quaternion with applied rotation around an axis.
    static Quaternion axisRotation(real alpha, Vector!(qt, 3) axis)
	{
        if(alpha == 0)
		{
            return Quaternion.identity;
        }
        Quaternion ret;

        alpha /= 2;
        qt sinaqt = to!qt(sin(alpha));

        ret.w = to!qt(cos(alpha));
        ret.x = axis.x * sinaqt;
        ret.y = axis.y * sinaqt;
        ret.z = axis.z * sinaqt;

        return ret;
    }

    /// Creates a quaternion from an euler rotation.
    static Quaternion eulerRotation(real heading, real attitude, real bank)
	{
        Quaternion ret;

        real c1 = cos(heading / 2);
        real s1 = sin(heading / 2);
        real c2 = cos(attitude / 2);
        real s2 = sin(attitude / 2);
        real c3 = cos(bank / 2);
        real s3 = sin(bank / 2);

        ret.w = to!qt(c1 * c2 * c3 - s1 * s2 * s3);
        ret.x = to!qt(s1 * s2 * c3 + c1 * c2 * s3);
        ret.y = to!qt(s1 * c2 * c3 + c1 * s2 * s3);
        ret.z = to!qt(c1 * s2 * c3 - s1 * c2 * s3);

        return ret;
    }

    /// Rotates the current quaternion around the x-axis and returns $(I this).
    Quaternion rotateX(real alpha)
	{
        this = xRotation(alpha) * this;
        return this;
    }

    /// Rotates the current quaternion around the y-axis and returns $(I this).
    Quaternion rotateY(real alpha)
	{
        this = yRotation(alpha) * this;
        return this;
    }

    /// Rotates the current quaternion around the z-axis and returns $(I this).
    Quaternion rotateZ(real alpha)
	{
        this = zRotation(alpha) * this;
        return this;
    }

    /// Rotates the current quaternion around an axis and returns $(I this).
    Quaternion rotateAxis(real alpha, Vector!(qt, 3) axis) {
        this = axisRotation(alpha, axis) * this;
        return this;
    }

    /// Applies an euler rotation to the current quaternion and returns $(I this).
    Quaternion rotateEuler(real heading, real attitude, real bank)
	{
        this = eulerRotation(heading, attitude, bank) * this;
        return this;
    }

    unittest
	{
        assert(quat.xRotation(PI).quaternion[1..4] == [1.0f, 0.0f, 0.0f]);
        assert(quat.yRotation(PI).quaternion[1..4] == [0.0f, 1.0f, 0.0f]);
        assert(quat.zRotation(PI).quaternion[1..4] == [0.0f, 0.0f, 1.0f]);
        assert((quat.xRotation(PI).w == quat.yRotation(PI).w) && (quat.yRotation(PI).w == quat.zRotation(PI).w));
        //assert(quat.rotateX(PI).w == to!(quat.qt)(cos(PI)));
        assert(quat.xRotation(PI).quaternion == quat.identity.rotateX(PI).quaternion);
        assert(quat.yRotation(PI).quaternion == quat.identity.rotateY(PI).quaternion);
        assert(quat.zRotation(PI).quaternion == quat.identity.rotateZ(PI).quaternion);

        assert(quat.axisRotation(PI, vec3(1.0f, 1.0f, 1.0f)).quaternion[1..4] == [1.0f, 1.0f, 1.0f]);
        assert(quat.axisRotation(PI, vec3(1.0f, 1.0f, 1.0f)).w == quat.xRotation(PI).w);
        assert(quat.axisRotation(PI, vec3(1.0f, 1.0f, 1.0f)).quaternion ==
               quat.identity.rotateAxis(PI, vec3(1.0f, 1.0f, 1.0f)).quaternion);

        quat q1 = quat.eulerRotation(PI, PI, PI);
        assert((q1.x > -1e-16) && (q1.x < 1e-16));
        assert((q1.y > -1e-16) && (q1.y < 1e-16));
        assert((q1.z > -1e-16) && (q1.z < 1e-16));
        assert(q1.w == -1.0f);
        assert(quat.eulerRotation(PI, PI, PI).quaternion == quat.identity.rotateEuler(PI, PI, PI).quaternion);
    }

    Quaternion opBinary(string op : "*")(Quaternion inp) const
	{
        Quaternion ret;

        ret.w = -x * inp.x - y * inp.y - z * inp.z + w * inp.w;
        ret.x = x * inp.w + y * inp.z - z * inp.y + w * inp.x;
        ret.y = -x * inp.z + y * inp.w + z * inp.x + w * inp.y;
        ret.z = x * inp.y - y * inp.x + z * inp.w + w * inp.z;

        return ret;
    }

    auto opBinaryRight(string op, T)(T inp) const if(!isQuaternion!T)
	{
        return this.opBinary!(op)(inp);
    }

    Quaternion opBinary(string op)(Quaternion inp) const
	if((op == "+") || (op == "-"))
	{
        Quaternion ret;

        mixin("ret.w = w" ~ op ~ "inp.w;");
        mixin("ret.x = x" ~ op ~ "inp.x;");
        mixin("ret.y = y" ~ op ~ "inp.y;");
        mixin("ret.z = z" ~ op ~ "inp.z;");

        return ret;
    }

    Vector!(qt, 3) opBinary(string op : "*")(Vector!(qt, 3) inp) const
	{
        Vector!(qt, 3) ret;

        qt ww = w^^2;
        qt w2 = w * 2;
        qt wx2 = w2 * x;
        qt wy2 = w2 * y;
        qt wz2 = w2 * z;
        qt xx = x^^2;
        qt x2 = x * 2;
        qt xy2 = x2 * y;
        qt xz2 = x2 * z;
        qt yy = y^^2;
        qt yz2 = 2 * y * z;
        qt zz = z * z;

        ret.vector =  [ww * inp.x + wy2 * inp.z - wz2 * inp.y + xx * inp.x +
                       xy2 * inp.y + xz2 * inp.z - zz * inp.x - yy * inp.x,
                       xy2 * inp.x + yy * inp.y + yz2 * inp.z + wz2 * inp.x -
                       zz * inp.y + ww * inp.y - wx2 * inp.z - xx * inp.y,
                       xz2 * inp.x + yz2 * inp.y + zz * inp.z - wy2 * inp.x -
                       yy * inp.z + wx2 * inp.y - xx * inp.z + ww * inp.z];

       return ret;
    }

    Quaternion opBinary(string op : "*")(qt inp) const
	{
        return Quaternion(w*inp, x*inp, y*inp, z*inp);
    }

    void opOpAssign(string op : "*")(Quaternion inp)
	{
        qt w2 = -x * inp.x - y * inp.y - z * inp.z + w * inp.w;
        qt x2 = x * inp.w + y * inp.z - z * inp.y + w * inp.x;
        qt y2 = -x * inp.z + y * inp.w + z * inp.x + w * inp.y;
        qt z2 = x * inp.y - y * inp.x + z * inp.w + w * inp.z;
        w = w2; x = x2; y = y2; z = z2;
    }

    void opOpAssign(string op)(Quaternion inp)
	if((op == "+") || (op == "-"))
	{
        mixin("w = w" ~ op ~ "inp.w;");
        mixin("x = x" ~ op ~ "inp.x;");
        mixin("y = y" ~ op ~ "inp.y;");
        mixin("z = z" ~ op ~ "inp.z;");
    }

    void opOpAssign(string op : "*")(qt inp)
	{
        quaternion[0] *= inp;
        quaternion[1] *= inp;
        quaternion[2] *= inp;
        quaternion[3] *= inp;
    }

    unittest
	{
        quat q1 = quat.identity;
        quat q2 = quat(3.0f, 0.0f, 1.0f, 2.0f);
        quat q3 = quat(3.4f, 0.1f, 1.2f, 2.3f);

        assert((q1 * q1).quaternion == q1.quaternion);
        assert((q1 * q2).quaternion == q2.quaternion);
        assert((q2 * q1).quaternion == q2.quaternion);
        quat q4 = q3 * q2;
        assert((q2 * q3).quaternion != q4.quaternion);
        q3 *= q2;
        assert(q4.quaternion == q3.quaternion);
        assert(almostEqual(q4.x, 0.4f));
        assert(almostEqual(q4.y, 6.8f));
        assert(almostEqual(q4.z, 13.8f));
        assert(almostEqual(q4.w, 4.4f));

        quat q5 = quat(1.0f, 2.0f, 3.0f, 4.0f);
        quat q6 = quat(3.0f, 1.0f, 6.0f, 2.0f);

        assert((q5 - q6).quaternion == [-2.0f, 1.0f, -3.0f, 2.0f]);
        assert((q5 + q6).quaternion == [4.0f, 3.0f, 9.0f, 6.0f]);
        assert((q6 - q5).quaternion == [2.0f, -1.0f, 3.0f, -2.0f]);
        assert((q6 + q5).quaternion == [4.0f, 3.0f, 9.0f, 6.0f]);
        q5 += q6;
        assert(q5.quaternion == [4.0f, 3.0f, 9.0f, 6.0f]);
        q6 -= q6;
        assert(q6.quaternion == [0.0f, 0.0f, 0.0f, 0.0f]);

        quat q7 = quat(2.0f, 2.0f, 2.0f, 2.0f);
        assert((q7 * 2).quaternion == [4.0f, 4.0f, 4.0f, 4.0f]);
        assert((2 * q7).quaternion == (q7 * 2).quaternion);
        q7 *= 2;
        assert(q7.quaternion == [4.0f, 4.0f, 4.0f, 4.0f]);

        vec3 v1 = vec3(1.0f, 2.0f, 3.0f);
        assert((q1 * v1).vector == v1.vector);
        assert((v1 * q1).vector == (q1 * v1).vector);
        assert((q2 * v1).vector == [-2.0f, 36.0f, 38.0f]);
    }

    const int opCmp(ref const Quaternion qua) const
	{
        foreach(i, a; quaternion)
		{
            if(a < qua.quaternion[i])
			{
                return -1;
            }
			else if(a > qua.quaternion[i])
			{
                return 1;
            }
        }

        // Quaternions are the same
        return 0;
    }

    bool opEquals(const Quaternion qu) const
	{
        return quaternion == qu.quaternion;
    }

    bool opCast(T : bool)() const
	{
        return isFinite;
    }

    unittest
	{
        assert(quat(1.0f, 2.0f, 3.0f, 4.0f) == quat(1.0f, 2.0f, 3.0f, 4.0f));
        assert(quat(1.0f, 2.0f, 3.0f, 4.0f) != quat(1.0f, 2.0f, 3.0f, 3.0f));

        assert(!(quat(float.nan, float.nan, float.nan, float.nan)));
        if(quat(1.0f, 1.0f, 1.0f, 1.0f)) { }
        else { assert(false); }
    }

}

/// Pre-defined quaternion of type float.
alias Quaternion!(float) quat;

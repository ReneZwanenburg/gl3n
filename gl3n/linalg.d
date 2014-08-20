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
import gl3n.util : isQuaternion, TupleRange;
import gl3n.vector;
import gl3n.matrix;


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

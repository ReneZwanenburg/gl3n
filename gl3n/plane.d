module gl3n.plane;

import gl3n.linalg : Vector, dot, homogeneousPoint;
import gl3n.math : almostEqual;
import gl3n.util : isVector;

import std.traits : isFloatingPoint;

/// Base template for all plane-types.
/// Params:
/// type = all values get stored as this type (must be floating point)
struct PlaneT(type = float)
if(isFloatingPoint!type)
{
	alias vt = Vector!(type, 4); /// Convenience alias to the corresponding vector type.

    vt p;

    @safe pure nothrow:

    /// Constructs the plane from a 4-dimensional vector
    this(vt p)
	{
        this.p = p;
    }

    unittest
	{
        auto p = PlaneT(vt(0.0f, 1.0f, 2.0f, 3.0f));
        assert(p.p == vt(0.0f, 1.0f, 2.0f, 3.0f));

        p.p.x = 4.0f;
        assert(p.normal == vt(4.0f, 1.0f, 2.0f));
        assert(p.a == 4.0f);
        assert(p.b == 1.0f);
        assert(p.c == 2.0f);
        assert(p.d == 3.0f);
    }

    /// Normalizes the plane inplace.
    void normalize()
	{
        p *= 1 / p.xyz.length;
    }

    /// Returns a normalized copy of the plane.
    @property PlaneT normalized() const
	{
        PlaneT ret = this;
        ret.normalize();
        return ret;
    }

    unittest
	{
        auto p = PlaneT(vt(0.0f, 1.0f, 2.0f), 3.0f);
        auto pn = p.normalized();
        assert(pn.normal == vec3(0.0f, 1.0f, 2.0f).normalized);
        assert(almostEqual(pn.d, 3.0f/vt(0.0f, 1.0f, 2.0f).length));
        p.normalize();
        assert(p == pn);
    }

    /// Returns the distance from a point to the plane.
    /// Note: the plane $(RED must) be normalized, the result can be negative.
    auto distance(T)(T point) const
	if(isVector!T)
	{
        return dot(point.homogeneousPoint, p);
    }

    /// Returns the distance from a point to the plane.
    /// Note: the plane does not have to be normalized, the result can be negative.
    auto ndistance(T)(T point) const
	if(isVector!T)
	{
		return normalized.distance(point);
    }

    unittest
	{
        auto p = PlaneT(vt(-1.0f, 4.0f, 19.0f), -10.0f);
        assert(almostEqual(p.ndistance(vt(5.0f, -2.0f, 0.0f)), -1.182992));
        assert(almostEqual(p.ndistance(vt(5.0f, -2.0f, 0.0f)),
                            p.normalized.distance(vt(5.0f, -2.0f, 0.0f))));
    }
}

alias PlaneT!(float) Plane;